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
package ai.djl.modality.rl;

import ai.djl.modality.rl.env.RlEnv.Step;

/**
 * Records {@link Step}s so that they can be trained on.
 *
 * <p>Using a replay buffer ensures that a variety of states are trained on for every training batch
 * making the training more stable.
 */
public interface ReplayBuffer {

    /**
     * Returns a batch of steps from this buffer.
     *
     * @return a batch of steps from this buffer
     */
    Step[] getBatch();

    /**
     * Adds a new step to the buffer.
     *
     * @param step the step to add
     */
    void addStep(Step step);
}
