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
package org.tensorflow.engine;

import java.lang.management.MemoryUsage;
import org.tensorflow.TensorFlow;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;

public class TfEngine extends Engine {

    TfEngine() {}

    @Override
    public Model newModel(Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return "Tensorflow";
    }

    /** {@inheritDoc} */
    @Override
    public int getGpuCount() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public MemoryUsage getGpuMemory(Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Context defaultContext() {
        return Context.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return TensorFlow.version();
    }

    @Override
    public GradientCollector newGradientCollector() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public ParameterStore newParameterStore(Optimizer optimizer, boolean aggregateOnGPU) {
        return null;
    }

    @Override
    public NDManager newBaseManager() {
        return TfNDManager.newBaseManager();
    }

    @Override
    public NDManager newBaseManager(Context context) {
        return TfNDManager.newBaseManager(context);
    }
}
