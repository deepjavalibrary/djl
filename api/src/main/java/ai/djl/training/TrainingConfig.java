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
import ai.djl.nn.Parameter;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.PairList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.function.Predicate;

/**
 * An interface that is responsible for holding the configuration required by {@link Trainer}.
 *
 * <p>A trainer requires different information to facilitate the training process. This information
 * is passed by using this configuration.
 *
 * <p>The required options for the configuration are:
 *
 * <ul>
 *   <li><b>Required</b> {@link Loss} - A loss function is used to measure how well a model matches
 *       the dataset. Because the lower value of the function is better, it is called the "loss"
 *       function. This is the only required configuration.
 *   <li>{@link Evaluator} - An evaluator is used to measure how well a model matches the dataset.
 *       Unlike the loss, they are only there for people to look at and are not used for
 *       optimization. Since many losses are not as intuitive, adding other evaluators can help to
 *       understand how the model is doing. We recommend adding as many as possible.
 *   <li>{@link Device} - The device is what hardware should be used to train your model on.
 *       Typically, this is either GPU or GPU. The default is to use a single GPU if it is available
 *       or CPU if not.
 *   <li>{@link Initializer} - The initializer is used to set the initial values of the model's
 *       parameters before training. This can usually be left as the default initializer.
 *   <li>{@link Optimizer} - The optimizer is the algorithm that updates the model parameters to
 *       minimize the loss function. There are a variety of optimizers, most of which are variants
 *       of stochastic gradient descent. When you are just starting, you can use the default
 *       optimizer. Later on, customizing the optimizer can result in faster training.
 *   <li>{@link ExecutorService} - The executorService is used for parallelization when training
 *       batches on multiple GPUs or loading data from the dataset. If none is provided, all
 *       operations with be sequential.
 *   <li>{@link TrainingListener} - The training listeners add additional functionality to the
 *       training process through a listener interface. This can include showing training progress,
 *       stopping early if the training fails, or recording performance metrics. We offer several
 *       easy sets of {@link TrainingListener.Defaults}.
 * </ul>
 */
public interface TrainingConfig {

    /**
     * Gets the {@link Device} that are available for computation.
     *
     * <p>This is necessary for a {@link Trainer} as it needs to know what kind of device it is
     * running on, and how many devices it is running on.
     *
     * @return an array of {@link Device}
     */
    Device[] getDevices();

    /**
     * Gets a list of {@link Initializer} and Predicate to initialize the parameters of the model.
     *
     * @return an {@link Initializer}
     */
    PairList<Initializer, Predicate<Parameter>> getInitializers();

    /**
     * Gets the {@link Optimizer} to use during training.
     *
     * @return an {@link Optimizer}
     */
    Optimizer getOptimizer();

    /**
     * Gets the {@link Loss} function to compute the loss against.
     *
     * @return a {@link Loss} function
     */
    Loss getLossFunction();

    /**
     * Gets the {@link ExecutorService} for parallelization.
     *
     * @return an {@link ExecutorService}
     */
    ExecutorService getExecutorService();

    /**
     * Returns the list of {@link Evaluator}s that should be computed during training.
     *
     * @return a list of {@link Evaluator}s
     */
    List<Evaluator> getEvaluators();

    /**
     * Returns the list of {@link TrainingListener}s that should be used during training.
     *
     * @return a list of {@link TrainingListener}s
     */
    List<TrainingListener> getTrainingListeners();
}
