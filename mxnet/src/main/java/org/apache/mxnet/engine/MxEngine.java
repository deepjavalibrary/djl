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
package org.apache.mxnet.engine;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.Profiler;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.training.Trainer;
import java.io.File;

public class MxEngine extends Engine {

    MxEngine() {}

    @Override
    public int getGpuCount() {
        return 0;
    }

    @Override
    public Context defaultContext() {
        return null;
    }

    @Override
    public String getVersion() {
        return null;
    }

    @Override
    public Model loadModel(File modelPath, String modelName, int epoch) {
        return null;
    }

    @Override
    public Predictor<NDArray, NDArray> newPredictor(Model model, Context context) {
        return new MxPredictor((MxModel) model, context);
    }

    @Override
    public Trainer newTrainer(Model model, Context context) {
        return null;
    }

    @Override
    public void setProfiler(Profiler profiler) {}

    @Override
    public NDFactory getNDFactory() {
        return null;
    }
}
