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

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDList;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Metadata;
import ai.djl.repository.Repository;
import ai.djl.repository.VersionRange;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/** Shared code for the {@link ModelLoader} implementations. */
public abstract class BaseModelLoader<I, O> implements ModelLoader<I, O> {

    protected Repository repository;
    protected MRL mrl;
    protected String version;
    protected Map<Pair<Type, Type>, TranslatorFactory<?, ?>> factories;
    protected ModelZoo modelZoo;

    private Metadata metadata;

    /**
     * Constructs a {@link ModelLoader} given the repository, mrl, and version.
     *
     * @param repository the repository to load the model from
     * @param mrl the mrl of the model to load
     * @param version the version of the model to load
     * @param modelZoo the modelZoo type that is being used to get supported engine types
     */
    protected BaseModelLoader(Repository repository, MRL mrl, String version, ModelZoo modelZoo) {
        this.repository = repository;
        this.mrl = mrl;
        this.version = version;
        factories = new ConcurrentHashMap<>();
        factories.put(
                new Pair<>(NDList.class, NDList.class),
                (TranslatorFactory<NDList, NDList>) arguments -> new NoopTranslator());
        this.modelZoo = modelZoo;
    }

    /** {@inheritDoc} */
    @Override
    public String getArtifactId() {
        return mrl.getArtifactId();
    }

    /** {@inheritDoc} */
    @Override
    public <S, T> ZooModel<S, T> loadModel(Criteria<S, T> criteria)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Artifact artifact = match(criteria.getFilters());
        if (artifact == null) {
            throw new ModelNotFoundException("Model not found.");
        }

        Map<String, Object> override = criteria.getArguments();
        Progress progress = criteria.getProgress();
        Map<String, Object> arguments = artifact.getArguments(override);

        try {
            Translator<S, T> translator = criteria.getTranslator();
            if (translator == null) {
                TranslatorFactory<S, T> factory = getTranslatorFactory(criteria);
                if (factory == null) {
                    throw new ModelNotFoundException("No matching default translator found.");
                }

                translator = factory.newInstance(arguments);
            }

            repository.prepare(artifact, progress);
            if (progress != null) {
                progress.reset("Loading", 2);
                progress.update(1);
            }

            Path modelPath = repository.getResourceDirectory(artifact);

            // Check if the engine is specified in Criteria, use it if it is.
            // Otherwise check the modelzoo supported engine and grab a random engine in the list.
            // Otherwise if none of them is specified or model zoo is null, go to default engine.
            String engine = criteria.getEngine();
            if (engine == null && modelZoo != null) {
                String defaultEngine = Engine.getInstance().getEngineName();
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
            if (engine != null && !Engine.hasEngine(engine)) {
                throw new ModelNotFoundException(engine + " is not supported.");
            }

            String modelName = criteria.getModelName();
            if (modelName == null) {
                modelName = artifact.getName();
            }

            Model model =
                    createModel(modelName, Device.defaultDevice(), artifact, arguments, engine);
            if (criteria.getBlock() != null) {
                model.setBlock(criteria.getBlock());
            }
            model.load(modelPath, null, criteria.getOptions());
            return new ZooModel<>(model, translator);
        } finally {
            if (progress != null) {
                progress.end();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<Artifact> listModels() throws IOException, ModelNotFoundException {
        List<Artifact> list = getMetadata().getArtifacts();
        return list.stream()
                .filter(a -> version == null || version.equals(a.getVersion()))
                .collect(Collectors.toList());
    }

    protected Model createModel(
            String name,
            Device device,
            Artifact artifact,
            Map<String, Object> arguments,
            String engine)
            throws IOException {
        return Model.newInstance(name, device, engine);
    }

    /**
     * Returns the first artifact that matches a given criteria.
     *
     * @param criteria the criteria to match against
     * @return the first artifact that matches the criteria. Null will be returned if no artifact
     *     matches
     * @throws IOException for errors while loading the model
     * @throws ModelNotFoundException if the metadata to get artifacts from is not found
     */
    protected Artifact match(Map<String, String> criteria)
            throws IOException, ModelNotFoundException {
        List<Artifact> list = search(criteria);
        if (list.isEmpty()) {
            return null;
        }
        return list.get(0);
    }

    /**
     * Returns all the artifacts that match a given criteria.
     *
     * @param criteria the criteria to match against
     * @return all the artifacts that match a given criteria
     * @throws IOException for errors while loading the model
     * @throws ModelNotFoundException if the metadata to get artifacts from is not found
     */
    private List<Artifact> search(Map<String, String> criteria)
            throws IOException, ModelNotFoundException {
        return getMetadata().search(VersionRange.parse(version), criteria);
    }

    private Metadata getMetadata() throws IOException, ModelNotFoundException {
        if (metadata == null) {
            metadata = repository.locate(mrl);
            if (metadata == null) {
                throw new ModelNotFoundException(mrl.getArtifactId() + " Models not found.");
            }
        }
        return metadata;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(repository.getName())
                .append(':')
                .append(mrl.getGroupId())
                .append(':')
                .append(mrl.getArtifactId())
                .append(" [\n");
        try {
            for (Artifact artifact : listModels()) {
                sb.append('\t').append(artifact).append('\n');
            }
        } catch (IOException | ModelNotFoundException e) {
            sb.append("\tFailed load metadata.");
        }
        sb.append("\n]");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private <S, T> TranslatorFactory<S, T> getTranslatorFactory(Criteria<S, T> criteria) {
        return (TranslatorFactory<S, T>)
                factories.get(new Pair<>(criteria.getInputClass(), criteria.getOutputClass()));
    }
}
