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
package com.amazon.ai.test.mock;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.Profiler;
import com.amazon.ai.Translator;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.nn.NNIndex;
import com.amazon.ai.training.Trainer;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.management.MemoryUsage;

public class MockEngine extends Engine {

    private int gpuCount;
    private MemoryUsage gpuMemory;
    private Context context;
    private String version;

    @Override
    public String getEngineName() {
        return "MockEngine";
    }

    @Override
    public int getGpuCount() {
        return gpuCount;
    }

    @Override
    public MemoryUsage getGpuMemory(Context context) {
        return gpuMemory;
    }

    @Override
    public Context defaultContext() {
        return context;
    }

    @Override
    public String getVersion() {
        return version;
    }

    @Override
    public Model loadModel(File modelPath, String modelName, int epoch) throws IOException {
        if (!modelPath.exists()) {
            throw new FileNotFoundException("File not found: " + modelPath);
        }
        return new MockModel();
    }

    @Override
    public <I, O> Predictor<I, O> newPredictor(
            Model model, Translator<I, O> translator, Context context) {
        return new MockPredictor<>(model, translator, context);
    }

    @Override
    public NNIndex getNNIndex() {
        return null;
    }

    @Override
    public Trainer newTrainer(Model model, Context context) {
        return null;
    }

    @Override
    public void setProfiler(Profiler profiler) {}

    public void setGpuCount(int gpuCount) {
        this.gpuCount = gpuCount;
    }

    public void setGpuMemory(MemoryUsage gpuMemory) {
        this.gpuMemory = gpuMemory;
    }

    public void setContext(Context context) {
        this.context = context;
    }

    public void setVersion(String version) {
        this.version = version;
    }
}
