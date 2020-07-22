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
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import java.util.List;

/**
 * An interface that is responsible for holding the configuration required by {@link Trainer}.
 *
 * <p>A {@link Trainer} requires an {@link Initializer} to initialize the parameters of the model,
 * an {@link Optimizer} to compute gradients and update the parameters according to a {@link Loss}
 * function. It also needs to know the {@link Evaluator}s that need to be computed during training.
 * A {@code TrainingConfig} instance that is passed to the {@link Trainer} will provide this
 * information, and thus facilitate the training process.
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
     * Gets the {@link Initializer} to initialize the parameters of the model.
     *
     * @return an {@link Initializer}
     */
    Initializer getInitializer();

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
     * Gets the {@link DataManager} that computes data and labels from the output of dataset.
     *
     * @return a {@link DataManager}
     */
    DataManager getDataManager();

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
