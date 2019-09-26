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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FactorTracker extends LearningRateTracker {

    private static final Logger logger = LoggerFactory.getLogger(FactorTracker.class);

    private float factor;
    private float stopFactorLearningRate;
    private float step;
    private int count;

    public FactorTracker(
            float baseLearningRate,
            int warmupSteps,
            float warmupBeginLearningRate,
            WarmupMode warmupMode,
            int step,
            float factor,
            float stopFactorLearningRate) {
        super(baseLearningRate, warmupSteps, warmupBeginLearningRate, warmupMode);
        this.step = step;
        this.factor = factor;
        this.stopFactorLearningRate = stopFactorLearningRate;
        this.count = 0;
    }

    @Override
    public float getNewLearningRate(int numUpdate) {
        if (numUpdate < warmupSteps) {
            return getWarmupLearningRate(numUpdate);
        }
        while (numUpdate > count + step) {
            count += step;
            baseLearningRate = factor;
            if (baseLearningRate < stopFactorLearningRate) {
                baseLearningRate = stopFactorLearningRate;
                logger.debug(
                        "Update[{}]: now learning rate arrived at {}, will not change in the future",
                        numUpdate,
                        String.format("%.5e", baseLearningRate));
            } else {
                logger.debug(
                        "Update[{}]: Change learning rate to {}",
                        numUpdate,
                        String.format("%.5e", baseLearningRate));
            }
        }
        return baseLearningRate;
    }
}
