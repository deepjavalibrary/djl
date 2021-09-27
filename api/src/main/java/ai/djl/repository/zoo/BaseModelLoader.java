/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.repository.zoo;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.translate.DefaultTranslatorFactory;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Collectors;

/** Shared code for the {@link ModelLoader} implementations. */
public class BaseModelLoader implements ModelLoader {

    protected MRL mrl;
    protected TranslatorFactory defaultFactory;

    /**
     * Constructs a {@link ModelLoader} given the repository, mrl, and version.
     *
     * @param mrl the mrl of the model to load
     */
    public BaseModelLoader(MRL mrl) {
        this.mrl = mrl;
        defaultFactory = new DefaultTranslatorFactory();
    }

    /** {@inheritDoc} */
    @Override
    public String getArtifactId() {
        return mrl.getArtifactId();
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return mrl.getApplication();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> ZooModel<I, O> loadModel(Criteria<I, O> criteria)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Artifact artifact = mrl.match(criteria.getFilters());
        if (artifact == null) {
            throw new ModelNotFoundException("No matching filter found");
        }

        Progress progress = criteria.getProgress();
        Map<String, Object> arguments = artifact.getArguments(criteria.getArguments());
        Map<String, String> options = artifact.getOptions(criteria.getOptions());

        try {
            TranslatorFactory factory = getTranslatorFactory(criteria, arguments);
            Class<I> input = criteria.getInputClass();
            Class<O> output = criteria.getOutputClass();
            if (factory == null || !factory.isSupported(input, output)) {
                factory = defaultFactory;
                if (!factory.isSupported(input, output)) {
                    throw new ModelNotFoundException(getFactoryLookupErrorMessage(factory));
                }
            }

            mrl.prepare(artifact, progress);
            if (progress != null) {
                progress.reset("Loading", 2);
                progress.update(1);
            }

            Path modelPath = mrl.getRepository().getResourceDirectory(artifact);
            Path modelDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (modelDir == null) {
                throw new AssertionError("Directory should not be null.");
            }

            loadServingProperties(modelDir, arguments, options);
            Application application = criteria.getApplication();
            if (application != Application.UNDEFINED) {
                arguments.put("application", application.getPath());
            }
            String engine = criteria.getEngine();
            if (engine == null) {
                // get engine from serving.properties
                engine = (String) arguments.get("engine");
            }

            // Check if the engine is specified in Criteria, use it if it is.
            // Otherwise check the modelzoo supported engine and grab a random engine in the list.
            // Otherwise if none of them is specified or model zoo is null, go to default engine.
            if (engine == null) {
                ModelZoo modelZoo = ModelZoo.getModelZoo(mrl.getGroupId());
                if (modelZoo != null) {
                    String defaultEngine = Engine.getDefaultEngineName();
                    for (String supportedEngine : modelZoo.getSupportedEngines()) {
                        if (supportedEngine.equals(defaultEngine)) {
                            engine = supportedEngine;
                            break;
                        } else if (Engine.hasEngine(supportedEngine)) {
                            engine = supportedEngine;
                        }
                    }
                    if (engine == null) {
                        throw new ModelNotFoundException(
                                "No supported engine available for model zoo: "
                                        + modelZoo.getGroupId());
                    }
                }
            }
            if (engine != null && !Engine.hasEngine(engine)) {
                throw new ModelNotFoundException(engine + " is not supported");
            }

            String modelName = criteria.getModelName();
            if (modelName == null) {
                modelName = artifact.getName();
            }

            Model model =
                    createModel(
                            modelDir,
                            modelName,
                            criteria.getDevice(),
                            criteria.getBlock(),
                            arguments,
                            engine);
            model.load(modelPath, null, options);
            Translator<I, O> translator =
                    (Translator<I, O>) factory.newInstance(input, output, model, arguments);
            return new ZooModel<>(model, translator);
        } catch (TranslateException e) {
            throw new ModelNotFoundException("No matching translator found", e);
        } finally {
            if (progress != null) {
                progress.end();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<Artifact> listModels() throws IOException {
        List<Artifact> list = mrl.listArtifacts();
        String version = mrl.getVersion();
        return list.stream()
                .filter(a -> version == null || version.equals(a.getVersion()))
                .collect(Collectors.toList());
    }

    protected Model createModel(
            Path modelPath,
            String name,
            Device device,
            Block block,
            Map<String, Object> arguments,
            String engine)
            throws IOException {
        Model model = Model.newInstance(name, device, engine);
        if (block == null) {
            String className = (String) arguments.get("blockFactory");
            BlockFactory factory = ClassLoaderUtils.findImplementation(modelPath, className);
            if (factory != null) {
                block = factory.newBlock(model, modelPath, arguments);
            }
        }
        if (block != null) {
            model.setBlock(block);
        }
        return model;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(mrl.getGroupId())
                .append(':')
                .append(mrl.getArtifactId())
                .append(' ')
                .append(getApplication())
                .append(" [\n");
        try {
            for (Artifact artifact : listModels()) {
                sb.append('\t').append(artifact).append('\n');
            }
        } catch (IOException e) {
            sb.append("\tFailed load metadata.");
        }
        sb.append(']');
        return sb.toString();
    }

    protected TranslatorFactory getTranslatorFactory(
            Criteria<?, ?> criteria, Map<String, Object> arguments) {
        TranslatorFactory factory = criteria.getTranslatorFactory();
        if (factory != null) {
            return factory;
        }

        String factoryClass = (String) arguments.get("translatorFactory");
        if (factoryClass != null) {
            ClassLoader cl = Thread.currentThread().getContextClassLoader();
            factory = ClassLoaderUtils.initClass(cl, factoryClass);
        }
        return factory;
    }

    private String getFactoryLookupErrorMessage(TranslatorFactory factory) {
        StringBuilder sb = new StringBuilder(200);
        sb.append(
                "No matching default translator found. The valid input and output classes are: \n");
        for (Pair<Type, Type> io : factory.getSupportedTypes()) {
            sb.append("\t(")
                    .append(io.getKey().getTypeName())
                    .append(", ")
                    .append(io.getValue().getTypeName())
                    .append(")\n");
        }
        return sb.toString();
    }

    private void loadServingProperties(
            Path modelDir, Map<String, Object> arguments, Map<String, String> options)
            throws IOException {
        Path manifestFile = modelDir.resolve("serving.properties");
        if (Files.isRegularFile(manifestFile)) {
            Properties prop = new Properties();
            try (Reader reader = Files.newBufferedReader(manifestFile)) {
                prop.load(reader);
            }
            for (String key : prop.stringPropertyNames()) {
                if (key.startsWith("option.")) {
                    options.putIfAbsent(key.substring(7), prop.getProperty(key));
                } else {
                    arguments.putIfAbsent(key, prop.getProperty(key));
                }
            }
        }
    }
}
