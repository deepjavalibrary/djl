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

/**
 * {@code MultiFactorTracker} is an implementation of {@link LearningRateTracker} which is updated
 * by a multiplicative factor, at an uneven interval of steps, until it reaches a specified stop
 * value.
 */
public class MultiFactorTracker extends LearningRateTracker {
    private static final Logger logger = LoggerFactory.getLogger(FactorTracker.class);

    private int[] steps;
    private float factor;
    private int stepIndex;

    /**
     * Creates a new instance of {@code MultiFactorTracker}.
     *
     * @param builder the builder to create a new instance of {@code MultiFactorTracker}
     */
    public MultiFactorTracker(Builder builder) {
        super(builder);
        this.steps = builder.steps;
        this.factor = builder.factor;
    }

    /** {@inheritDoc} */
    @Override
    public float getNewLearningRate(int numUpdate) {
        if (numUpdate < warmUpSteps) {
            return getWarmUpLearningRate(numUpdate);
        }
        while (stepIndex <= steps.length - 1) {
            if (numUpdate > steps[stepIndex]) {
                stepIndex++;
                baseLearningRate *= factor;
                logger.debug(
                        "Update[{}]: Change learning rate to {}",
                        numUpdate,
                        String.format("%.5e", baseLearningRate));
            } else {
                checkLearningRate(baseLearningRate);
                return baseLearningRate;
            }
        }
        checkLearningRate(baseLearningRate);
        return baseLearningRate;
    }

    /** The Builder to construct an {@link MultiFactorTracker} object. */
    public static final class Builder extends LrBaseBuilder<Builder> {
        private int[] steps;
        private float factor = 1;

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets an array of integers indicating when the learning rate should be changed, usually in
         * an uneven interval of steps.
         *
         * @param steps an array of integers indicating when the learning rate should be change
         * @return this {@code Builder}
         */
        public Builder setSteps(int[] steps) {
            if (steps.length <= 1) {
                throw new IllegalArgumentException(
                        "Steps should be an array of integers indicating when the "
                                + "learning rate should be changed, usually in an uneven interval of steps"
                                + "use FactorTracker if you want learning rate to be changed at a constant interval of steps");
            }
            for (int i = 0; i < steps.length; i++) {
                if (i > 0 && steps[i] <= steps[i - 1]) {
                    throw new IllegalArgumentException("Steps must be an increasing list");
                }
                if (steps[i] < 1) {
                    throw new IllegalArgumentException("Step must be larger or equal to 1");
                }
            }
            this.steps = steps;
            return this;
        }

        /**
         * Set the value of the multiplicative factor.
         *
         * @param factor the value of the multiplicative factor
         * @return this {@code Builder}
         */
        public Builder optFactor(float factor) {
            if (factor > 1f) {
                throw new IllegalArgumentException("factor should be no more than 1");
            }
            this.factor = factor;
            return this;
        }

        /**
         * Builds a {@link MultiFactorTracker} block.
         *
         * @return the {@link MultiFactorTracker} block
         */
        public MultiFactorTracker build() {
            if (steps == null) {
                throw new IllegalArgumentException("Steps must be set to change learning rate");
            }
            return new MultiFactorTracker(this);
        }
    }
}
