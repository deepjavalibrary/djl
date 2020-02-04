/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.hyperparameter.param;

import ai.djl.training.hyperparameter.optimizer.HpOptimizer;

/**
 * A class representing an input to the network that can't be differentiated.
 *
 * <p>Some hyperparameters include learning rates, network sizes and shapes, activation choices, and
 * model selection. In order to evaluate a set of hyperparameters, the only way is to fully train
 * your model using those choices of hyperparameters. So, the full training loop involves training
 * the model a number of times using different choices of hyperparameters. This can be mostly
 * automated by using a {@link HpOptimizer}.
 *
 * @param <T> the type of the hyperparameter
 */
public abstract class Hyperparameter<T> {

    protected String name;

    /**
     * Constructs a hyperparameter with the given name.
     *
     * @param name the name of the hyperparameter
     */
    public Hyperparameter(String name) {
        this.name = name;
    }

    /**
     * Returns the name of the hyperparameter.
     *
     * @return the name of the hyperparameter
     */
    public String getName() {
        return name;
    }

    /**
     * Returns a random value for the hyperparameter for a range of a fixed value if it is a {@link
     * HpVal}.
     *
     * @return a random value for the hyperparameter for a range of a fixed value if it is a {@link
     *     HpVal}
     */
    public abstract T random();
}
