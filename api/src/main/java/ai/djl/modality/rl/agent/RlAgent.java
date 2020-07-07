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
package ai.djl.modality.rl.agent;

import ai.djl.modality.rl.env.RlEnv;
import ai.djl.modality.rl.env.RlEnv.Step;
import ai.djl.ndarray.NDList;

/**
 * An {@link RlAgent} is the model or technique to decide the actions to take in an {@link RlEnv}.
 */
public interface RlAgent {

    /**
     * Chooses the next action to take within the {@link RlEnv}.
     *
     * @param env the current environment
     * @param training true if the agent is currently traning
     * @return the action to take
     */
    NDList chooseAction(RlEnv env, boolean training);

    /**
     * Trains this {@link RlAgent} on a batch of {@link Step}s.
     *
     * @param batchSteps the steps to train on
     */
    void trainBatch(Step[] batchSteps);
}
