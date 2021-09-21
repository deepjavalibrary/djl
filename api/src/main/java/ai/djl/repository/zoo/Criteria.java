/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.nn.Block;
import ai.djl.translate.DefaultTranslatorFactory;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code Criteria} class contains search criteria to look up a {@link ZooModel}.
 *
 * <p>Criteria follows Builder pattern. See {@link Builder} for detail. In DJL's builder convention,
 * the methods start with {@code set} are required fields, and {@code opt} for optional fields.
 *
 * <p>Examples
 *
 * <pre>
 * Criteria&lt;Image, Classifications&gt; criteria = Criteria.builder()
 *         .setTypes(Image.class, Classifications.class) // defines input and output data type
 *         .optTranslator(ImageClassificationTranslator.builder().setSynsetArtifactName("synset.txt").build())
 *         .optModelUrls("file:///var/models/my_resnet50") // search models in specified path
 *         .optModelName("resnet50") // specify model file prefix
 *         .build();
 * </pre>
 *
 * <p>See <a href="http://docs.djl.ai/docs/load_model.html#criteria-class">Model loading</a> for
 * more detail.
 *
 * @param <I> the model input type
 * @param <O> the model output type
 */
public class Criteria<I, O> {

    private Application application;
    private Class<I> inputClass;
    private Class<O> outputClass;
    private String engine;
    private Device device;
    private String groupId;
    private String artifactId;
    private ModelZoo modelZoo;
    private Map<String, String> filters;
    private Map<String, Object> arguments;
    private Map<String, String> options;
    private TranslatorFactory factory;
    private Block block;
    private String modelName;
    private Progress progress;

    Criteria(Builder<I, O> builder) {
        this.application = builder.application;
        this.inputClass = builder.inputClass;
        this.outputClass = builder.outputClass;
        this.engine = builder.engine;
        this.device = builder.device;
        this.groupId = builder.groupId;
        this.artifactId = builder.artifactId;
        this.modelZoo = builder.modelZoo;
        this.filters = builder.filters;
        this.arguments = builder.arguments;
        this.options = builder.options;
        this.factory = builder.factory;
        this.block = builder.block;
        this.modelName = builder.modelName;
        this.progress = builder.progress;
    }

    /**
     * Load the {@link ZooModel} that matches this criteria.
     *
     * @return the model that matches the criteria
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<I, O> loadModel()
            throws IOException, ModelNotFoundException, MalformedModelException {
        Logger logger = LoggerFactory.getLogger(ModelZoo.class);
        logger.debug("Loading model with {}", this);

        List<ModelZoo> list = new ArrayList<>();
        if (modelZoo != null) {
            logger.debug("Searching model in specified model zoo: {}", modelZoo.getGroupId());
            if (groupId != null && !modelZoo.getGroupId().equals(groupId)) {
                throw new ModelNotFoundException(
                        "groupId conflict with ModelZoo criteria."
                                + modelZoo.getGroupId()
                                + " v.s. "
                                + groupId);
            }
            Set<String> supportedEngine = modelZoo.getSupportedEngines();
            if (engine != null && !supportedEngine.contains(engine)) {
                throw new ModelNotFoundException(
                        "ModelZoo doesn't support specified engine: " + engine);
            }
            list.add(modelZoo);
        } else {
            for (ModelZoo zoo : ModelZoo.listModelZoo()) {
                if (groupId != null && !zoo.getGroupId().equals(groupId)) {
                    // filter out ModelZoo by groupId
                    logger.debug("Ignore ModelZoo {} by groupId: {}", zoo.getGroupId(), groupId);
                    continue;
                }
                Set<String> supportedEngine = zoo.getSupportedEngines();
                if (engine != null && !supportedEngine.contains(engine)) {
                    logger.debug("Ignore ModelZoo {} by engine: {}", zoo.getGroupId(), engine);
                    continue;
                }
                list.add(zoo);
            }
        }

        Exception lastException = null;
        for (ModelZoo zoo : list) {
            String loaderGroupId = zoo.getGroupId();
            for (ModelLoader loader : zoo.getModelLoaders()) {
                Application app = loader.getApplication();
                String loaderArtifactId = loader.getArtifactId();
                logger.debug("Checking ModelLoader: {}", loader);
                if (artifactId != null && !artifactId.equals(loaderArtifactId)) {
                    // filter out by model loader artifactId
                    logger.debug(
                            "artifactId mismatch for ModelLoader: {}:{}",
                            loaderGroupId,
                            loaderArtifactId);
                    continue;
                }
                if (application != Application.UNDEFINED
                        && app != Application.UNDEFINED
                        && !app.matches(application)) {
                    // filter out ModelLoader by application
                    logger.debug(
                            "application mismatch for ModelLoader: {}:{}",
                            loaderGroupId,
                            loaderArtifactId);
                    continue;
                }

                try {
                    return loader.loadModel(this);
                } catch (ModelNotFoundException e) {
                    lastException = e;
                    logger.trace("", e);
                    logger.debug(
                            "{} for ModelLoader: {}:{}",
                            e.getMessage(),
                            loaderGroupId,
                            loaderArtifactId);
                }
            }
        }
        throw new ModelNotFoundException(
                "No matching model with specified Input/Output type found.", lastException);
    }

    /**
     * Returns the application of the model.
     *
     * @return the application of the model
     */
    public Application getApplication() {
        return application;
    }

    /**
     * Returns the input data type.
     *
     * @return the input data type
     */
    public Class<I> getInputClass() {
        return inputClass;
    }

    /**
     * Returns the output data type.
     *
     * @return the output data type
     */
    public Class<O> getOutputClass() {
        return outputClass;
    }

    /**
     * Returns the engine name.
     *
     * @return the engine name
     */
    public String getEngine() {
        return engine;
    }

    /**
     * Returns the {@link Device} of the model to be loaded on.
     *
     * @return the {@link Device} of the model to be loaded on
     */
    public Device getDevice() {
        return device;
    }

    /**
     * Returns the groupId of the {@link ModelZoo} to be searched.
     *
     * @return the groupId of the {@link ModelZoo} to be searched
     */
    public String getGroupId() {
        return groupId;
    }

    /**
     * Returns the artifactId of the {@link ModelLoader} to be searched.
     *
     * @return the artifactIds of the {@link ModelLoader} to be searched
     */
    public String getArtifactId() {
        return artifactId;
    }

    /**
     * Returns the {@link ModelZoo} to be searched.
     *
     * @return the {@link ModelZoo} to be searched
     */
    public ModelZoo getModelZoo() {
        return modelZoo;
    }

    /**
     * Returns the search filters that must match the properties of the model.
     *
     * @return the search filters that must match the properties of the model.
     */
    public Map<String, String> getFilters() {
        return filters;
    }

    /**
     * Returns the override configurations of the model loading arguments.
     *
     * @return the override configurations of the model loading arguments
     */
    public Map<String, Object> getArguments() {
        return arguments;
    }

    /**
     * Returns the model loading options.
     *
     * @return the model loading options
     */
    public Map<String, String> getOptions() {
        return options;
    }

    /**
     * Returns the optional {@link TranslatorFactory} to be used for {@link ZooModel}.
     *
     * @return the optional {@link TranslatorFactory} to be used for {@link ZooModel}
     */
    public TranslatorFactory getTranslatorFactory() {
        return factory;
    }

    /**
     * Returns the optional {@link Block} to be used for {@link ZooModel}.
     *
     * @return the optional {@link Block} to be used for {@link ZooModel}
     */
    public Block getBlock() {
        return block;
    }

    /**
     * Returns the optional model name to be used for {@link ZooModel}.
     *
     * @return the optional model name to be used for {@link ZooModel}
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns the optional {@link Progress} for the model loading.
     *
     * @return the optional {@link Progress} for the model loading
     */
    public Progress getProgress() {
        return progress;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append("Criteria:\n");
        if (application != null) {
            sb.append("\tApplication: ").append(application).append('\n');
        }
        sb.append("\tInput: ").append(inputClass);
        sb.append("\n\tOutput: ").append(outputClass).append('\n');
        if (engine != null) {
            sb.append("\tEngine: ").append(engine).append('\n');
        }
        if (modelZoo != null) {
            sb.append("\tModelZoo: ").append(modelZoo.getGroupId()).append('\n');
        }
        if (groupId != null) {
            sb.append("\tGroupID: ").append(groupId).append('\n');
        }
        if (artifactId != null) {
            sb.append("\tArtifactId: ").append(artifactId).append('\n');
        }
        if (filters != null) {
            sb.append("\tFilter: ").append(JsonUtils.GSON.toJson(filters)).append('\n');
        }
        if (arguments != null) {
            sb.append("\tArguments: ").append(JsonUtils.GSON.toJson(arguments)).append('\n');
        }
        if (options != null) {
            sb.append("\tOptions: ").append(JsonUtils.GSON.toJson(options)).append('\n');
        }
        if (factory == null) {
            sb.append("\tNo translator supplied\n");
        }
        return sb.toString();
    }

    /**
     * Creates a builder to build a {@code Criteria}.
     *
     * <p>The methods start with {@code set} are required fields, and {@code opt} for optional
     * fields.
     *
     * @return a new builder
     */
    public static Builder<?, ?> builder() {
        return new Builder<>();
    }

    /** A Builder to construct a {@code Criteria}. */
    public static final class Builder<I, O> {

        Application application;
        Class<I> inputClass;
        Class<O> outputClass;
        String engine;
        Device device;
        String groupId;
        String artifactId;
        ModelZoo modelZoo;
        Map<String, String> filters;
        Map<String, Object> arguments;
        Map<String, String> options;
        TranslatorFactory factory;
        Block block;
        String modelName;
        Progress progress;
        private Translator<I, O> translator;

        Builder() {
            application = Application.UNDEFINED;
        }

        private Builder(Class<I> inputClass, Class<O> outputClass, Builder<?, ?> parent) {
            this.inputClass = inputClass;
            this.outputClass = outputClass;
            application = parent.application;
            engine = parent.engine;
            device = parent.device;
            groupId = parent.groupId;
            filters = parent.filters;
            arguments = parent.arguments;
            options = parent.options;
            block = parent.block;
            modelName = parent.modelName;
            progress = parent.progress;
        }

        /**
         * Creates a new @{code Builder} class with the specified input and output data type.
         *
         * @param <P> the input data type
         * @param <Q> the output data type
         * @param inputClass the input class
         * @param outputClass the output class
         * @return a new @{code Builder} class with the specified input and output data type
         */
        public <P, Q> Builder<P, Q> setTypes(Class<P> inputClass, Class<Q> outputClass) {
            return new Builder<>(inputClass, outputClass, this);
        }

        /**
         * Sets the model application for this criteria.
         *
         * @param application the model application
         * @return this {@code Builder}
         */
        public Builder<I, O> optApplication(Application application) {
            this.application = application;
            return this;
        }

        /**
         * Sets the engine name for this criteria.
         *
         * @param engine the engine name
         * @return this {@code Builder}
         */
        public Builder<I, O> optEngine(String engine) {
            this.engine = engine;
            return this;
        }

        /**
         * Sets the {@link Device} for this criteria.
         *
         * @param device the {@link Device} for the criteria
         * @return this {@code Builder}
         */
        public Builder<I, O> optDevice(Device device) {
            this.device = device;
            return this;
        }

        /**
         * Sets optional groupId of the {@link ModelZoo} for this criteria.
         *
         * @param groupId the groupId of the {@link ModelZoo}
         * @return this {@code Builder}
         */
        public Builder<I, O> optGroupId(String groupId) {
            this.groupId = groupId;
            return this;
        }

        /**
         * Sets optional artifactId of the {@link ModelLoader} for this criteria.
         *
         * @param artifactId the artifactId of the {@link ModelLoader}
         * @return this {@code Builder}
         */
        public Builder<I, O> optArtifactId(String artifactId) {
            if (artifactId != null && artifactId.contains(":")) {
                String[] tokens = artifactId.split(":", -1);
                groupId = tokens[0].isEmpty() ? null : tokens[0];
                this.artifactId = tokens[1].isEmpty() ? null : tokens[1];
            } else {
                this.artifactId = artifactId;
            }
            return this;
        }

        /**
         * Sets optional model urls of the {@link ModelLoader} for this criteria.
         *
         * @param modelUrls the comma delimited url string
         * @return this {@code Builder}
         */
        public Builder<I, O> optModelUrls(String modelUrls) {
            if (modelUrls != null) {
                this.modelZoo = new DefaultModelZoo(modelUrls);
            }
            return this;
        }

        /**
         * Sets the optional model path of the {@link ModelLoader} for this criteria.
         *
         * @param modelPath the path to the model folder/files
         * @return this {@code Builder}
         */
        public Builder<I, O> optModelPath(Path modelPath) {
            if (modelPath != null) {
                try {
                    this.modelZoo = new DefaultModelZoo(modelPath.toUri().toURL().toString());
                } catch (MalformedURLException e) {
                    throw new AssertionError("Invalid model path: " + modelPath, e);
                }
            }
            return this;
        }

        /**
         * Sets optional {@link ModelZoo} of the {@link ModelLoader} for this criteria.
         *
         * @param modelZoo ModelZoo} of the {@link ModelLoader} for this criteria
         * @return this {@code Builder}
         */
        public Builder<I, O> optModelZoo(ModelZoo modelZoo) {
            this.modelZoo = modelZoo;
            return this;
        }

        /**
         * Sets the extra search filters for this criteria.
         *
         * @param filters the extra search filters
         * @return this {@code Builder}
         */
        public Builder<I, O> optFilters(Map<String, String> filters) {
            this.filters = filters;
            return this;
        }

        /**
         * Sets an extra search filter for this criteria.
         *
         * @param key the search key
         * @param value the search value
         * @return this {@code Builder}
         */
        public Builder<I, O> optFilter(String key, String value) {
            if (filters == null) {
                filters = new HashMap<>();
            }
            filters.put(key, value);
            return this;
        }

        /**
         * Sets an optional model {@link Block} for this criteria.
         *
         * @param block optional model {@link Block} for this criteria
         * @return this {@code Builder}
         */
        public Builder<I, O> optBlock(Block block) {
            this.block = block;
            return this;
        }

        /**
         * Sets an optional model name for this criteria.
         *
         * @param modelName optional model name for this criteria
         * @return this {@code Builder}
         */
        public Builder<I, O> optModelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Sets an extra model loading argument for this criteria.
         *
         * @param arguments optional model loading arguments
         * @return this {@code Builder}
         */
        public Builder<I, O> optArguments(Map<String, Object> arguments) {
            this.arguments = arguments;
            return this;
        }

        /**
         * Sets the optional model loading argument for this criteria.
         *
         * @param key the model loading argument key
         * @param value the model loading argument value
         * @return this {@code Builder}
         */
        public Builder<I, O> optArgument(String key, Object value) {
            if (arguments == null) {
                arguments = new HashMap<>();
            }
            arguments.put(key, value);
            return this;
        }

        /**
         * Sets the model loading options for this criteria.
         *
         * @param options the model loading options
         * @return this {@code Builder}
         */
        public Builder<I, O> optOptions(Map<String, String> options) {
            this.options = options;
            return this;
        }

        /**
         * Sets the optional model loading option for this criteria.
         *
         * @param key the model loading option key
         * @param value the model loading option value
         * @return this {@code Builder}
         */
        public Builder<I, O> optOption(String key, String value) {
            if (options == null) {
                options = new HashMap<>();
            }
            options.put(key, value);
            return this;
        }

        /**
         * Sets the optional {@link Translator} to override default {@code Translator}.
         *
         * @param translator the override {@code Translator}
         * @return this {@code Builder}
         */
        public Builder<I, O> optTranslator(Translator<I, O> translator) {
            this.translator = translator;
            return this;
        }

        /**
         * Sets the optional {@link TranslatorFactory} to override default {@code Translator}.
         *
         * @param factory the override {@code TranslatorFactory}
         * @return this {@code Builder}
         */
        public Builder<I, O> optTranslatorFactory(TranslatorFactory factory) {
            this.factory = factory;
            return this;
        }

        /**
         * Set the optional {@link Progress}.
         *
         * @param progress the {@code Progress}
         * @return this {@code Builder}
         */
        public Builder<I, O> optProgress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /**
         * Builds a {@link Criteria} instance.
         *
         * @return the {@link Criteria} instance
         */
        public Criteria<I, O> build() {
            if (factory == null && translator != null) {
                DefaultTranslatorFactory f = new DefaultTranslatorFactory();
                f.registerTranslator(inputClass, outputClass, translator);
                factory = f;
            }
            return new Criteria<>(this);
        }
    }
}
