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
import com.amazon.ai.nn.NNIndex;
import com.amazon.ai.training.Trainer;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
     * Get the name of the Engine
     *
     * @return
     */
    public abstract String getEngineName();

    /**
     * Get the initialized Engine
     *
     * @return Engine
     */
    public static Engine getInstance() {
        return ENGINE;
    }

    /**
     * Get the number of GPU in the system
     *
     * @return number of GPUs
     */
    public abstract int getGpuCount();

    /**
     * Default context specified by the system If GPU > 0, then default context is gpu(0) Otherwise,
     * the context will be cpu()
     *
     * @return default context
     */
    public abstract Context defaultContext();

    /**
     * Get the version of the Deep Learning Framework
     *
     * @return version number
     */
    public abstract String getVersion();

    /**
     * DO NOT USE THIS! use mode.loadModel instead Load the model passed from the model class
     * Preliminary check on the model path and name to see if the file exist if the file exist, will
     * handover to the corresponding Framework model loader
     *
     * @param modelPath Directory of the model
     * @param modelName Name/Prefix of the model
     * @param epoch Number of epoch of the model
     * @return Model contains the model information
     * @throws IOException Exception for file loading
     */
    public abstract Model loadModel(File modelPath, String modelName, int epoch) throws IOException;

    /**
     * DO NOT USE THIS! use Predictor.newInstance instead Create new predictor with specific Engine
     *
     * @param model the model used for inference
     * @param translator preprocessing and postprocessing helper class
     * @param context context to work on inference
     * @param <I> Input Object for the Predictor
     * @param <O> Output Object for the Predicor
     * @return Predictor
     */
    public abstract <I, O> Predictor<I, O> newPredictor(
            Model model, Translator<I, O> translator, Context context);

    /**
     * DO NOT USE THIS!
     *
     * @return The index of Neural Network operators to create a Block
     */
    public abstract NNIndex getNNIndex();

    /**
     * DO NOT USE THIS! use Trainer.newInstance() instead Load the model and create a Trainer to
     * starting training process
     *
     * @param model the model created to train on
     * @param context the context of training, can be CPU/GPU
     * @return Trainer
     */
    public abstract Trainer newTrainer(Model model, Context context);

    public abstract void setProfiler(Profiler profiler);
}
