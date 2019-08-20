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

public class FactorTracker extends LrTracker {

    private static final Logger logger = LoggerFactory.getLogger(FactorTracker.class);

    private float factor;
    private float stopFactorLr;
    private float step;
    private int count;

    public FactorTracker(
            float baseLr,
            int warmupSteps,
            float warmupBeginLr,
            WarmupMode warmupMode,
            int step,
            float factor,
            float stopFactorLR) {
        super(baseLr, warmupSteps, warmupBeginLr, warmupMode);
        this.step = step;
        this.factor = factor;
        this.stopFactorLr = stopFactorLR;
        this.count = 0;
    }

    @Override
    public float getNewLearningRate(int numUpdate) {
        if (numUpdate < warmupSteps) {
            return getWarmupLr(numUpdate);
        }
        while (numUpdate > count + step) {
            count += step;
            baseLr = factor;
            if (baseLr < stopFactorLr) {
                baseLr = stopFactorLr;
                logger.debug(
                        "Update[%d]: now learning rate arrived at %.5e, will not change in the future",
                        numUpdate, baseLr);
            } else {
                logger.debug("Update[%d]: Change learning rate to %.5e", numUpdate, baseLr);
            }
        }
        return baseLr;
    }
}
