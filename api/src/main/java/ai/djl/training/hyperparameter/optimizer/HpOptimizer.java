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
package ai.djl.training.hyperparameter.optimizer;

import ai.djl.training.hyperparameter.param.HpSet;
import ai.djl.training.hyperparameter.param.Hyperparameter;
import ai.djl.util.Pair;

/**
 * An optimizer for {@link Hyperparameter}s.
 *
 * @see Hyperparameter
 */
public interface HpOptimizer {

    /**
     * Returns the next hyperparameters to test.
     *
     * @return the hyperparameters to test
     */
    HpSet nextConfig();

    /**
     * Updates the optimizer with the results of a hyperparameter test.
     *
     * @param config the tested hyperparameters
     * @param loss the <b>validation</b> loss from training with those hyperparameters
     */
    void update(HpSet config, float loss);

    /**
     * Returns the recorded loss.
     *
     * @param config the hyperparameters that were trained with
     * @return the loss
     * @throws java.util.NoSuchElementException if the hyperparameters were not trained with before
     */
    float getLoss(HpSet config);

    /**
     * Returns the best hyperparameters and loss.
     *
     * @return the best hyperparameters and loss
     */
    Pair<HpSet, Float> getBest();
}
