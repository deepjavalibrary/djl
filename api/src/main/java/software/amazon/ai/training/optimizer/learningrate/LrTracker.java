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
package software.amazon.ai.training.optimizer.learningrate;

public abstract class LrTracker {

    // TODO: Add abstraction on Joule level
    float baseLr;
    int warmupSteps;
    float warmupBeginLr;
    float warmupFinalLr;
    WarmupMode warmupMode;

    /**
     * A tracker returns a new learning rate based on the number of updates that have been
     * performed.
     *
     * @param baseLR The initial learning rate
     * @param warmupSteps number of warmup steps used before this scheduler starts decay
     * @param warmupBeginLr if using warmup, the learning rate from which it starts warming up
     * @param warmupMode warmup can be done in two modes. 'linear' mode gradually increases lr with
     *     each step in equal increments 'constant' mode keeps lr at warmup_begin_lr for
     *     warmup_steps
     */
    LrTracker(float baseLR, int warmupSteps, float warmupBeginLr, WarmupMode warmupMode) {
        this.baseLr = baseLR;
        this.warmupSteps = warmupSteps;
        this.warmupBeginLr = warmupBeginLr;
        this.warmupMode = warmupMode;
        this.warmupFinalLr = baseLR;
    }

    float getWarmupLr(int numUpdate) {
        if (warmupMode == WarmupMode.LINEAR) {
            return warmupBeginLr + (warmupFinalLr - warmupBeginLr) * numUpdate / warmupSteps;
        }
        return warmupBeginLr;
    }

    public abstract float getNewLearningRate(int numUpdate);

    public static final LrTracker fixedLR(float lr) {
        return new FixedLr(lr);
    }
}
