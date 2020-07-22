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
import ai.djl.training.tracker.Tracker;
import ai.djl.util.RandomUtils;

/**
 * The {@link EpsilonGreedy} is a simple exploration/excitation agent.
 *
 * <p>It helps other agents explore their environments during training by sometimes picking random
 * actions.
 *
 * <p>If a model based agent is used, it will only explore paths through the environment that have
 * already been seen. While this is sometimes good, it is also important to sometimes explore new
 * paths as well. This agent exhibits a tradeoff that takes random paths a fixed percentage of the
 * time during training.
 */
public class EpsilonGreedy implements RlAgent {

    private RlAgent baseAgent;
    private Tracker exploreRate;

    private int counter;

    /**
     * Constructs an {@link EpsilonGreedy}.
     *
     * @param baseAgent the (presumably model-based) agent to use for exploitation and to train
     * @param exploreRate the probability of taking a random action
     */
    public EpsilonGreedy(RlAgent baseAgent, Tracker exploreRate) {
        this.baseAgent = baseAgent;
        this.exploreRate = exploreRate;
    }

    /** {@inheritDoc} */
    @Override
    public NDList chooseAction(RlEnv env, boolean training) {
        if (training && RandomUtils.random() < exploreRate.getNewValue(counter++)) {
            return env.getActionSpace().randomAction();
        }
        return baseAgent.chooseAction(env, training);
    }

    /** {@inheritDoc} */
    @Override
    public void trainBatch(Step[] batchSteps) {
        baseAgent.trainBatch(batchSteps);
    }
}
