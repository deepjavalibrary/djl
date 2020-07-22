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
package ai.djl.training.tracker;

import ai.djl.util.Preconditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code FactorTracker} is an implementation of {@link Tracker} which is updated by a
 * multiplicative factor, at a constant interval of steps, until it reaches a specified stop value.
 */
public class FactorTracker extends Tracker {

    private static final Logger logger = LoggerFactory.getLogger(FactorTracker.class);

    private int step;
    private float factor;
    private float stopFactorValue;
    private int count;

    /**
     * Creates a new instance of {@code FactorTracker}.
     *
     * @param builder the builder to create a new instance of {@code FactorTracker}
     */
    public FactorTracker(Builder builder) {
        super(builder);
        this.step = builder.step;
        this.factor = builder.factor;
        this.stopFactorValue = builder.stopFactorValue;
        this.count = 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getNewValue(int numUpdate) {
        if (numUpdate < warmUpSteps) {
            return getWarmUpValue(numUpdate);
        }
        while (numUpdate > count + step) {
            count += step;
            baseValue *= factor;
            if (baseValue < stopFactorValue) {
                baseValue = stopFactorValue;
                logger.debug(
                        "Update[{}]: now tracker value arrived at {}, will not change in the future",
                        numUpdate,
                        String.format("%.5e", baseValue));
            } else {
                logger.debug(
                        "Update[{}]: Change tracker value to {}",
                        numUpdate,
                        String.format("%.5e", baseValue));
            }
        }
        checkValue(baseValue);
        return baseValue;
    }

    /** The Builder to construct an {@link FactorTracker} object. */
    public static final class Builder extends TrackerBaseBuilder<Builder> {

        int step;
        float factor = 1;
        float stopFactorValue = 1e-8f;

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the number of steps after which the multiplicative factor must be applied once.
         *
         * @param step the number of steps after which the multiplicative factor must be applied
         *     once
         * @return this {@code Builder}
         */
        public Builder setStep(int step) {
            if (step < 1) {
                throw new IllegalArgumentException("step should be larger or equal to 1");
            }
            this.step = step;
            return this;
        }

        /**
         * Sets the value of the multiplicative factor.
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
         * Sets the stop value after which the value should remain constant.
         *
         * @param stopFactorValue the stop value after which the value should remain constant
         * @return this {@code Builder}
         */
        public Builder optStopFactorValue(float stopFactorValue) {
            this.stopFactorValue = stopFactorValue;
            return this;
        }

        /**
         * Builds a {@link FactorTracker} block.
         *
         * @return the {@link FactorTracker} block
         */
        public FactorTracker build() {
            Preconditions.checkArgument(step > 0, "Step must be set to change value every N steps");
            return new FactorTracker(this);
        }
    }
}
