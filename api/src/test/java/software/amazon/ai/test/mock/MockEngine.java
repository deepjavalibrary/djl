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

import java.lang.management.MemoryUsage;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;

public class MockEngine extends Engine {

    private int gpuCount;
    private MemoryUsage gpuMemory;
    private Device device = Device.cpu();
    private String version;

    @Override
    public Model newModel(Device device) {
        return new MockModel(device);
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
    public MemoryUsage getGpuMemory(Device device) {
        return gpuMemory;
    }

    @Override
    public Device defaultDevice() {
        return device;
    }

    @Override
    public String getVersion() {
        return version;
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
    public NDManager newBaseManager(Device device) {
        return new MockNDManager();
    }

    public void setGpuCount(int gpuCount) {
        this.gpuCount = gpuCount;
    }

    public void setGpuMemory(MemoryUsage gpuMemory) {
        this.gpuMemory = gpuMemory;
    }

    public void setDevice(Device device) {
        this.device = device;
    }

    public void setVersion(String version) {
        this.version = version;
    }
}
