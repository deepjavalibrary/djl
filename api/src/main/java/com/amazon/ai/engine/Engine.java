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
package com.amazon.ai.engine;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.Profiler;
import com.amazon.ai.Translator;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.EngineUtils;
import com.amazon.ai.nn.NNIndex;
import com.amazon.ai.training.Trainer;
import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The <code>Engine</code> interface shadows difference between each deep learning frameworks.
 *
 * <p>Any framework specific functionality should be provided through this class.
 */
public abstract class Engine {

    private static final Logger logger = LoggerFactory.getLogger(Engine.class);

    private static final Engine ENGINE = initEngine();

    private static synchronized Engine initEngine() {
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        List<EngineProvider> list = new ArrayList<>();
        for (EngineProvider provider : loaders) {
            list.add(provider);
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
     * Get the name of the Engine.
     *
     * @return name of the engine
     */
    public abstract String getEngineName();

    /**
     * Get the initialized Engine.
     *
     * @return instance of <code>Engine</code>
     */
    public static Engine getInstance() {
        return ENGINE;
    }

    /**
     * Get the number of GPU in the system.
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
     * <p>If the system has GPU available, then default context is {@link
     * com.amazon.ai.Context#gpu()}, otherwise returns {@link Context#cpu()}
     *
     * @return default context
     */
    public abstract Context defaultContext();

    /**
     * Get the version of the Deep Learning Framework.
     *
     * @return version number
     */
    public abstract String getVersion();

    /**
     * Load the model passed from the model class.
     *
     * <p>We recommend to use {@link Model#loadModel(String, int)}. Preliminary check on the model
     * path and name to see if the file exist. If the file exist, will handover to the corresponding
     * Framework model loader
     *
     * @param modelPath Directory of the model
     * @param modelName Name/Prefix of the model
     * @param epoch Number of epoch of the model
     * @return {@link Model} contains the model information
     * @throws IOException Exception for file loading
     */
    public abstract Model loadModel(Path modelPath, String modelName, int epoch) throws IOException;

    /**
     * Create new predictor with specific Engine.
     *
     * <p>Recommend to use {@link Predictor#newInstance(Model, Translator, Context)}.
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
     * com.amazon.ai.nn}.
     *
     * @return The index of Neural Network operators to create a Block
     */
    public abstract NNIndex getNNIndex();

    /**
     * An internal helper to get the Engine specific implementations for utilities
     *
     * @return The engine specific utilities
     */
    public abstract EngineUtils getEngineUtils();

    /**
     * Try to use {@link Trainer}.newInstance() instead Load the model and create a Trainer to
     * starting training process
     *
     * @param model the model created to train on
     * @param context the context of training, can be CPU/GPU
     * @return Trainer
     */
    public abstract Trainer newTrainer(Model model, Context context);

    // TODO: Not Implemented
    public abstract void setProfiler(Profiler profiler);
}
