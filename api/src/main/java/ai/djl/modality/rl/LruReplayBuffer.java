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
import ai.djl.util.RandomUtils;

/**
 * A simple {@link ReplayBuffer} that randomly selects across the whole buffer, but always removes
 * the oldest items in the buffer once it is full.
 */
public class LruReplayBuffer implements ReplayBuffer {

    private int batchSize;

    private Step[] steps;
    private int firstStepIndex;
    private int stepsActualSize;

    /**
     * Constructs a {@link LruReplayBuffer}.
     *
     * @param batchSize the number of steps to train on per batch
     * @param bufferSize the number of steps to hold in the buffer
     */
    public LruReplayBuffer(int batchSize, int bufferSize) {
        this.batchSize = batchSize;
        steps = new Step[bufferSize];
        firstStepIndex = 0;
        stepsActualSize = 0;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidArrayLoops")
    public Step[] getBatch() {
        Step[] batch = new Step[batchSize];
        for (int i = 0; i < batchSize; i++) {
            int baseIndex = RandomUtils.nextInt(stepsActualSize);
            int index = Math.floorMod(firstStepIndex + baseIndex, steps.length);
            batch[i] = steps[index];
        }
        return batch;
    }

    /** {@inheritDoc} */
    @Override
    public void addStep(Step step) {
        if (stepsActualSize == steps.length) {
            int stepToReplace = Math.floorMod(firstStepIndex - 1, steps.length);
            steps[stepToReplace].close();
            steps[stepToReplace] = step;
            firstStepIndex = Math.floorMod(firstStepIndex + 1, steps.length);
        } else {
            steps[stepsActualSize] = step;
            stepsActualSize++;
        }
    }
}
