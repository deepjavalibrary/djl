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
package ai.djl.tensorflow.engine;

import java.lang.management.MemoryUsage;
import org.tensorflow.TensorFlow;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDManager;

public class TfEngine extends Engine {

    TfEngine() {}

    @Override
    public Model newModel(Device device) {
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
    public MemoryUsage getGpuMemory(Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Device defaultDevice() {
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return TensorFlow.version();
    }

    @Override
    public NDManager newBaseManager() {
        return TfNDManager.newBaseManager();
    }

    @Override
    public NDManager newBaseManager(Device device) {
        return TfNDManager.newBaseManager(device);
    }
}
