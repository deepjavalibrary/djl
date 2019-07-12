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
package software.amazon.ai.engine;

import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.Translator;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.NNIndex;
import software.amazon.ai.training.Trainer;

/**
 * The {@code Engine} interface shadows differences between each deep learning framework.
 *
 * <p>Any framework specific functionality should be provided through this class.
 */
public abstract class Engine {

    private static final Logger logger = LoggerFactory.getLogger(Engine.class);

    private static final Map<String, Engine> ALL_ENGINES = new ConcurrentHashMap<>();

    private static final Engine ENGINE = initEngine();

    private static synchronized Engine initEngine() {
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        List<EngineProvider> list = new ArrayList<>();
        for (EngineProvider provider : loaders) {
            list.add(provider);
            Engine engine = provider.getEngine();
            ALL_ENGINES.put(engine.getEngineName(), engine);
        }

        if (list.isEmpty()) {
            throw new EngineException("No deep learning engine found in class path.");
        }

        if (list.size() > 1) {
            logger.warn("More than one deep learning engines found.");
        }

        Engine engine = list.get(0).getEngine();
        logger.info("Loading ML engine from: {}", engine.getClass());
        return engine;
    }

    /**
     * Returns the name of the Engine.
     *
     * @return name of the engine
     */
    public abstract String getEngineName();

    /**
     * Returns the default Engine.
     *
     * @return instance of {@code Engine}
     */
    public static Engine getInstance() {
        return ENGINE;
    }

    /**
     * Returns {@code Engine} by engine name.
     *
     * @param engineName name of Engine to retrieve
     * @return instance of {@code Engine}
     */
    public static Engine getEngine(String engineName) {
        return ALL_ENGINES.get(engineName);
    }

    /**
     * Returns the number of GPUs available in the system.
     *
     * @return number of GPUs available in the system
     */
    public abstract int getGpuCount();

    /**
     * Returns {@link MemoryUsage} of specified GPU context.
     *
     * @param context the GPU {@link Context} to retrieve
     * @return {@link MemoryUsage} of specified GPU context
     * @throws EngineException if operation is not supported
     * @throws IllegalArgumentException if Context is not GPU context
     */
    public abstract MemoryUsage getGpuMemory(Context context);

    /**
     * Returns system default context.
     *
     * <p>If the system has GPU available, then the default context is {@link
     * software.amazon.ai.Context#gpu()}. Otherwise the default context returned is {@link
     * Context#cpu()}
     *
     * @return default context
     */
    public abstract Context defaultContext();

    /**
     * Returns the version of the deep learning framework.
     *
     * @return version number
     */
    public abstract String getVersion();

    /**
     * Loads the model from the specified location.
     *
     * <p>The model format is deep learning framework specific, each framework may have its own
     * loading options. You should check each framework's document for available loading options.
     *
     * @param modelPath Directory of the model
     * @param modelName Name/Prefix of the model
     * @param options load model options, check document for specific engine
     * @return {@link Model} contains the model information
     * @throws IOException Exception for file loading
     */
    public abstract Model loadModel(Path modelPath, String modelName, Map<String, String> options)
            throws IOException;

    /**
     * Creates new {@link Predictor} instance for this Engine.
     *
     * @param model the model used for inference
     * @param translator preprocessing and postprocessing helper class
     * @param context context to work on inference
     * @param <I> Input Object for the Predictor
     * @param <O> Output Object for the Predictor
     * @return Predictor
     */
    public abstract <I, O> Predictor<I, O> newPredictor(
            Model model, Translator<I, O> translator, Context context);

    /**
     * An internal helper to get the Engine specific implementations for the blocks in {@link
     * software.amazon.ai.nn}.
     *
     * @return The index of Neural Network operators to create a Block
     */
    public abstract NNIndex getNNIndex();

    /**
     * Creates a new {@link Trainer} instance for this Engine.
     *
     * @param model the model created to train on
     * @param context the context of training, can be CPU/GPU
     * @return Trainer
     */
    public abstract Trainer newTrainer(Model model, Context context);

    /**
     * Creates a new top-level {@link NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Context}.
     *
     * @return Returns a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager();

    /**
     * Creates a new top-level {@link NDManager} with specified {@link Context}.
     *
     * @param context default {@link Context}
     * @return Returns a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager(Context context);
}
