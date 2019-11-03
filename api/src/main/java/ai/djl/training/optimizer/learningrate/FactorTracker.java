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
package ai.djl.training.optimizer.learningrate;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FactorTracker extends LearningRateTracker {

    private static final Logger logger = LoggerFactory.getLogger(FactorTracker.class);

    private int step;
    private float factor;
    private float stopFactorLearningRate;
    private int count;

    public FactorTracker(Builder builder) {
        super(builder);
        this.step = builder.getStep();
        this.factor = builder.getFactor();
        this.stopFactorLearningRate = builder.getStopFactorLearningRate();
        this.count = 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getNewLearningRate(int numUpdate) {
        if (numUpdate < warmUpSteps) {
            return getWarmUpLearningRate(numUpdate);
        }
        while (numUpdate > count + step) {
            count += step;
            baseLearningRate *= factor;
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
        checkLearningRate(baseLearningRate);
        return baseLearningRate;
    }

    public static final class Builder extends LrBaseBuilder<Builder> {
        private int step;
        private float factor = 1;
        private float stopFactorLearningRate = 1e-8f;

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        public Builder setStep(int step) {
            if (step < 1) {
                throw new IllegalArgumentException("step should be larger or equal than 1");
            }
            this.step = step;
            return this;
        }

        public Builder optFactor(float factor) {
            if (factor > 1f) {
                throw new IllegalArgumentException("factor should be no more than 1");
            }
            this.factor = factor;
            return this;
        }

        public Builder optStopFactorLearningRate(float stopFactorLearningRate) {
            this.stopFactorLearningRate = stopFactorLearningRate;
            return this;
        }

        public float getStopFactorLearningRate() {
            return stopFactorLearningRate;
        }

        public int getStep() {
            return step;
        }

        public float getFactor() {
            return factor;
        }

        public FactorTracker build() {
            if (step == 0) {
                throw new IllegalArgumentException(
                        "Step must be set to change learning rate every N steps");
            }
            return new FactorTracker(this);
        }
    }
}
