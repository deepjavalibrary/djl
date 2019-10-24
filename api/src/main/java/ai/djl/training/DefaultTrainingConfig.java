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
import ai.djl.training.metrics.TrainingMetric;
import ai.djl.training.optimizer.Optimizer;
import java.util.ArrayList;
import java.util.List;

/** {@code DefaultTrainingConfig} is an implementation of the {@link TrainingConfig} interface. */
public class DefaultTrainingConfig implements TrainingConfig {

    private Initializer initializer;
    private Optimizer optimizer;
    private Device[] devices;
    private Loss loss;
    private List<TrainingMetric> trainingMetrics;
    private int batchSize;

    /**
     * Creates an instance of {@code DefaultTrainingConfig} with the given {@link Initializer}.
     *
     * @param initializer the initializer to initialize the parameters with
     * @param loss the loss to use for training
     */
    public DefaultTrainingConfig(Initializer initializer, Loss loss) {
        this.initializer = initializer;
        trainingMetrics = new ArrayList<>();
        this.loss = loss;
    }

    /**
     * Sets the array of {@link Device} available for training.
     *
     * @param devices an array of devices to be set
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig setDevices(Device[] devices) {
        this.devices = devices;
        return this;
    }

    /**
     * Sets the {@link Optimizer} used during training.
     *
     * @param optimizer the optimizer to be set
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    /**
     * Adds a {@link TrainingMetric} that needs to be computed during training.
     *
     * @param trainingMetric the training metric to be added
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig addTrainingMetric(TrainingMetric trainingMetric) {
        trainingMetrics.add(trainingMetric);
        return this;
    }

    /**
     * Sets the size of a batch for training.
     *
     * @param batchSize the batch size
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    /** {@inheritDoc} */
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

    /** {@inheritDoc} */
    @Override
    public Initializer getInitializer() {
        return initializer;
    }

    /** {@inheritDoc} */
    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }

    /** {@inheritDoc} */
    @Override
    public Loss getLossFunction() {
        return loss;
    }

    /** {@inheritDoc} */
    @Override
    public List<TrainingMetric> getTrainingMetrics() {
        return trainingMetrics;
    }

    /** {@inheritDoc} */
    @Override
    public int getBatchSize() {
        return batchSize;
    }
}
