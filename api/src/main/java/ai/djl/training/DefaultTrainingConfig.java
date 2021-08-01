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
import ai.djl.nn.Parameter;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Predicate;

/** {@code DefaultTrainingConfig} is an implementation of the {@link TrainingConfig} interface. */
public class DefaultTrainingConfig implements TrainingConfig {

    private PairList<Initializer, Predicate<Parameter>> initializers = new PairList<>();
    private Optimizer optimizer;
    private Device[] devices;
    private Loss loss;
    private ExecutorService executorService;
    private List<Evaluator> evaluators;
    private List<TrainingListener> listeners;

    /**
     * Creates an instance of {@code DefaultTrainingConfig} with the given {@link Loss}. {@code
     * DefaultTrainingConfig} creates a default {@link TrainingConfig}, {@link Adam} as optimiser,
     * and the given {@link Loss}. The evaluators and listeners are left to the user's discretion.
     *
     * @param loss the loss to use for training
     */
    public DefaultTrainingConfig(Loss loss) {
        this.loss = loss;
        optimizer = Adam.builder().build();
        evaluators = new ArrayList<>();
        listeners = new ArrayList<>();
    }

    /**
     * Sets the {@link Initializer} to use for the parameters (default from <a
     * href="https://arxiv.org/abs/1502.01852">paper</a>).
     *
     * @param initializer the initialer to use for the parameters
     * @param type the {@link Parameter.Type} of the parameters
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optInitializer(Initializer initializer, Parameter.Type type) {
        initializers.add(initializer, parameter -> parameter.getType().equals(type));
        return this;
    }

    /**
     * Sets the {@link Initializer} to use for the parameters (default from <a
     * href="https://arxiv.org/abs/1502.01852">paper</a>).
     *
     * @param initializer the initialer to use for the parameters
     * @param name the name of the parameter
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optInitializer(Initializer initializer, String name) {
        initializers.add(initializer, parameter -> parameter.getName().equals(name));
        return this;
    }

    /**
     * Sets the {@link Initializer} to use for the parameters (default from <a
     * href="https://arxiv.org/abs/1502.01852">paper</a>).
     *
     * @param initializer the initialer to use for the parameters
     * @param predicate the predicate to identify parameter
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optInitializer(
            Initializer initializer, Predicate<Parameter> predicate) {
        initializers.add(initializer, predicate);
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
     * Sets the {@link ExecutorService} with the global {@link ForkJoinPool#commonPool()}.
     *
     * @return this {@link DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optExecutorService() {
        return optExecutorService(ForkJoinPool.commonPool());
    }

    /**
     * Sets the {@link ExecutorService} to train with multiple threads.
     *
     * @param executorService the executor service
     * @return this {@link DefaultTrainingConfig}
     */
    public DefaultTrainingConfig optExecutorService(ExecutorService executorService) {
        this.executorService = executorService;
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
     * Adds {@link TrainingListener}s for training.
     *
     * @param listeners the {@link TrainingListener}s to add
     * @return this {@code DefaultTrainingConfig}
     */
    public DefaultTrainingConfig addTrainingListeners(TrainingListener... listeners) {
        this.listeners.addAll(Arrays.asList(listeners));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Device[] getDevices() {
        if (devices == null) {
            return Engine.getInstance().getDevices();
        }
        return devices;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<Initializer, Predicate<Parameter>> getInitializers() {
        return initializers;
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

    @Override
    public ExecutorService getExecutorService() {
        return executorService;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getEvaluators() {
        return evaluators;
    }

    /** {@inheritDoc} */
    @Override
    public List<TrainingListener> getTrainingListeners() {
        return listeners;
    }
}
