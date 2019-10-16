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
package ai.djl.training;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.TrainingMetrics;
import ai.djl.training.optimizer.Optimizer;
import java.util.ArrayList;
import java.util.List;

public class DefaultTrainingConfig implements TrainingConfig {

    private Initializer initializer;
    private Optimizer optimizer;
    private Device[] devices;
    private Loss loss;
    private List<TrainingMetrics> trainingMetrics;

    public DefaultTrainingConfig(Initializer initializer) {
        this.initializer = initializer;
        trainingMetrics = new ArrayList<>();
    }

    public DefaultTrainingConfig setDevices(Device[] devices) {
        this.devices = devices;
        return this;
    }

    public DefaultTrainingConfig setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    public DefaultTrainingConfig setLoss(Loss loss) {
        this.loss = loss;
        return this;
    }

    public DefaultTrainingConfig addTrainingMetrics(TrainingMetrics trainingMetrics) {
        this.trainingMetrics.add(trainingMetrics);
        return this;
    }

    @Override
    public Device[] getDevices() {
        if (devices == null) {
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

    @Override
    public Loss getLossFunction() {
        return loss;
    }

    @Override
    public List<TrainingMetrics> getTrainingMetrics() {
        return trainingMetrics;
    }
}
