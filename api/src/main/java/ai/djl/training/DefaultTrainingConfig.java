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
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.initializer.XavierInitializer.FactorType;
import ai.djl.training.initializer.XavierInitializer.RandomType;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import java.util.ArrayList;
import java.util.List;

/** {@code DefaultTrainingConfig} is an implementation of the {@link TrainingConfig} interface. */
public class DefaultTrainingConfig implements TrainingConfig {

    private Initializer initializer;
    private Optimizer optimizer;
    private Device[] devices;
    private Loss loss;
    private List<Evaluator> evaluators;
    private int batchSize;

    /**
     * Creates an instance of {@code DefaultTrainingConfig} with the given {@link Initializer}.
     *
     * @param loss the loss to use for training
     */
    public DefaultTrainingConfig(Loss loss) {
        // Defaults to initializer defined in https://arxiv.org/abs/1502.01852
        this.initializer = new XavierInitializer(RandomType.GAUSSIAN, FactorType.IN, 2);
        optimizer = new Adam.Builder().build();
        this.loss = loss;
        evaluators = new ArrayList<>();
    }

    /**
     * Sets the {@link Initializer} to use for the parameters (default from <a
     * href="https://arxiv.org/abs/1502.01852">paper</a>).
     *
     * @param initializer the initialer to use for the parameters
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    /**
     * Sets the array of {@link Device} available for training.
     *
     * @param devices an array of devices to be set
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optDevices(Device[] devices) {
        this.devices = devices;
        return this;
    }

    /**
     * Sets the {@link Optimizer} used during training (default {@link Adam}).
     *
     * @param optimizer the optimizer to be set
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    /**
     * Adds an {@link Evaluator} that needs to be computed during training.
     *
     * @param evaluator the evaluator to be added
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig addEvaluator(Evaluator evaluator) {
        evaluators.add(evaluator);
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
            return Device.getDevices(Integer.MAX_VALUE);
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
    public List<Evaluator> getEvaluators() {
        return evaluators;
    }

    /** {@inheritDoc} */
    @Override
    public int getBatchSize() {
        return batchSize;
    }
}
