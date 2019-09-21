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
    private boolean overwriteInitializer;

    public DefaultTrainingConfig(Initializer initializer, boolean overwriteInitializer) {
        this(initializer, overwriteInitializer, null);
    }

    public DefaultTrainingConfig(
            Initializer initializer, boolean overwriteInitializer, Optimizer optimizer) {
        this.initializer = initializer;
        this.overwriteInitializer = overwriteInitializer;
        this.optimizer = optimizer;
    }

    @Override
    public Device[] getDevices() {
        int numGpus = Engine.getInstance().getGpuCount();
        if (numGpus <= 0) {
            return new Device[] {Device.cpu()};
        }

        Device[] devices = new Device[numGpus];
        for (int i = 0; i < numGpus; i++) {
            devices[i] = Device.gpu(i);
        }
        return devices;
    }

    @Override
    public Initializer getInitializer() {
        return initializer;
    }

    @Override
    public boolean isOverwriteInitializer() {
        return overwriteInitializer;
    }

    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }
}
