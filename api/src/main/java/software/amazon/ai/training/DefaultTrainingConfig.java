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
package software.amazon.ai.training;

import software.amazon.ai.Device;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.optimizer.Optimizer;

public class DefaultTrainingConfig implements TrainingConfig {

    private Initializer initializer;
    private Optimizer optimizer;
    private Device[] devices;

    public DefaultTrainingConfig(Initializer initializer) {
        this(initializer, null);
    }

    public DefaultTrainingConfig(Initializer initializer, Optimizer optimizer) {
        this.initializer = initializer;
        this.optimizer = optimizer;
        int numGpus = Engine.getInstance().getGpuCount();
        if (numGpus > 0) {
            // TODO: Use single GPU by default for now.
            numGpus = 1;
            devices = new Device[numGpus];
            for (int i = 0; i < numGpus; i++) {
                devices[i] = Device.gpu(i);
            }
        } else {
            devices = new Device[] {Device.cpu()};
        }
    }

    public DefaultTrainingConfig(Initializer initializer, Optimizer optimizer, Device[] devices) {
        this.initializer = initializer;
        this.optimizer = optimizer;
        this.devices = devices;
    }

    @Override
    public Device[] getDevices() {
        return devices;
    }

    @Override
    public Initializer getInitializer() {
        return initializer;
    }

    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }
}
