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
import com.amazon.ai.Transformer;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.training.Trainer;
import java.io.File;
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

        return list.get(0).getEngine();
    }

    public static Engine getInstance() {
        return ENGINE;
    }

    public abstract int getGpuCount();

    public abstract Context defaultContext();

    public abstract String getVersion();

    public abstract Model loadModel(File modelPath, String modelName, int epoch);

    public abstract <I, O> Predictor<I, O> newPredictor(
            Model model, Transformer<I, O> transformer, Context context);

    public abstract Trainer newTrainer(Model model, Context context);

    public abstract void setProfiler(Profiler profiler);

    public abstract NDFactory getNDFactory();
}
