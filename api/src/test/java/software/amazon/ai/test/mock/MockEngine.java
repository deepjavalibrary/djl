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
package software.amazon.ai.test.mock;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;

public class MockEngine extends Engine {

    private int gpuCount;
    private MemoryUsage gpuMemory;
    private Context context = Context.cpu();
    private String version;

    @Override
    public Model newModel(Context context) {
        return new MockModel();
    }

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
    public Model loadModel(
            Path modelPath, String modelName, Context context, Map<String, String> options)
            throws IOException {
        if (Files.notExists(modelPath)) {
            throw new FileNotFoundException("File not found: " + modelPath);
        }
        return new MockModel();
    }

    @Override
    public GradientCollector newGradientCollector() {
        return null;
    }

    @Override
    public ParameterStore newParameterStore(Optimizer optimizer, boolean aggregateOnGPU) {
        return null;
    }

    @Override
    public NDManager newBaseManager() {
        return new MockNDManager();
    }

    @Override
    public NDManager newBaseManager(Context context) {
        return new MockNDManager();
    }

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
