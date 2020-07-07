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
package ai.djl.modality.rl.env;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/** An environment to use for reinforcement learning. */
public interface RlEnv extends AutoCloseable {

    /** Resets the environment to it's default state. */
    void reset();

    /**
     * Returns the observation detailing the current state of the environment.
     *
     * @return the observation detailing the current state of the environment
     */
    NDList getObservation();

    /**
     * Returns the current actions that can be taken in the environment.
     *
     * @return the current actions that can be taken in the environment
     */
    ActionSpace getActionSpace();

    /**
     * Takes a step by performing an action in this environment.
     *
     * @param action the action to perform
     * @param training true if the step is during training
     * @return the {@link Step} with the result of the action
     */
    Step step(NDList action, boolean training);

    /**
     * Runs the environment from reset until done.
     *
     * @param agent the agent to choose the actions with
     * @param training true to run while training. When training, the steps will be recorded
     * @return the total reward
     */
    default float runEnvironment(RlAgent agent, boolean training) {
        float totalReward = 0;
        reset();

        while (true) {
            NDList action = agent.chooseAction(this, training);
            Step step = step(action, training);
            totalReward += step.getReward().getFloat();
            if (step.isDone()) {
                return totalReward;
            }
        }
    }

    /**
     * Returns a batch of steps from the environment {@link ai.djl.modality.rl.ReplayBuffer}.
     *
     * @return a batch of steps from the environment {@link ai.djl.modality.rl.ReplayBuffer}
     */
    Step[] getBatch();

    /** {@inheritDoc} */
    @Override
    void close();

    /** A record of taking a step in the environment. */
    interface Step extends AutoCloseable {

        /**
         * Returns the observation detailing the state before the action.
         *
         * @return the observation detailing the state before the action
         */
        NDList getPreObservation();

        /**
         * Returns the action taken.
         *
         * @return the action taken
         */
        NDList getAction();

        /**
         * Returns the observation detailing the state after the action.
         *
         * @return the observation detailing the state after the action
         */
        NDList getPostObservation();

        /**
         * Returns the available actions after the step.
         *
         * @return the available actions after the step
         */
        ActionSpace getPostActionSpace();

        /**
         * Returns the reward given for the action.
         *
         * @return the reward given for the action
         */
        NDArray getReward();

        /**
         * Returns whether the environment is finished or can accept further actions.
         *
         * @return true if the environment is finished and can no longer accept further actions.
         */
        boolean isDone();

        /** {@inheritDoc} */
        @Override
        void close();
    }
}
