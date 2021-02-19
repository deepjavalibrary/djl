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

/**
 * {@code FactorTracker} is an implementation of {@link Tracker} which is updated by a
 * multiplicative factor.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/lr-scheduler.html#factor-tracker">For
 *     tracking learning rates, this section in the D2L chapter on learning rate scheduling</a>
 */
public class FactorTracker implements Tracker {

    private float baseValue;
    private float factor;
    private int maxUpdates;

    /**
     * Creates a new instance of {@code FactorTracker}.
     *
     * @param builder the builder to create a new instance of {@code FactorTracker}
     */
    public FactorTracker(Builder builder) {
        this.baseValue = builder.baseValue;
        this.factor = builder.factor;
        this.maxUpdates = builder.maxUpdates;
    }

    /**
     * Creates a new builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public float getNewValue(int numUpdate) {
        if (numUpdate > maxUpdates) {
            numUpdate = maxUpdates;
        }
        return (float) (baseValue * Math.pow(factor, numUpdate));
    }

    /** The Builder to construct an {@link FactorTracker} object. */
    public static final class Builder {

        private float baseValue;
        private float factor;
        Float min;
        Integer maxUpdates;

        private Builder() {}

        /**
         * Sets the initial value after no steps.
         *
         * @param baseValue the initial value
         * @return this {@code Builder}
         */
        public Builder setBaseValue(float baseValue) {
            this.baseValue = baseValue;
            return this;
        }

        /**
         * Sets the value of the multiplicative factor.
         *
         * @param factor the value of the multiplicative factor
         * @return this {@code Builder}
         */
        public Builder setFactor(float factor) {
            if (factor > 1f) {
                throw new IllegalArgumentException("factor should be no more than 1");
            }
            this.factor = factor;
            return this;
        }

        /**
         * Sets the minimum value.
         *
         * <p>This is equivalent to the max updates. Only one can be set.
         *
         * @param min the minimum value
         * @return this {@code Builder}
         */
        public Builder optMinValue(float min) {
            this.min = min;
            return this;
        }

        /**
         * Sets the maximum number of updates after which the value should remain constant.
         *
         * @param maxUpdates the maximum number of updates after which the value should remain
         *     constant
         * @return this {@code Builder}
         */
        public Builder optMaxUpdates(int maxUpdates) {
            this.maxUpdates = maxUpdates;
            return this;
        }

        /**
         * Builds a {@link FactorTracker} block.
         *
         * @return the {@link FactorTracker} block
         */
        public FactorTracker build() {
            Preconditions.checkArgument(factor != 0, "You must set a factor");

            if (min != null) {
                Preconditions.checkArgument(
                        maxUpdates == null, "You can not set both maxUpdates and a min value");
                Preconditions.checkArgument(
                        min < baseValue, "The min must be smaller than the base value");
                maxUpdates =
                        Math.toIntExact(Math.round(Math.log(min / baseValue) / Math.log(factor)));
            } else if (maxUpdates == null) {
                // Default to no max if none set
                maxUpdates = Integer.MAX_VALUE;
            }

            return new FactorTracker(this);
        }
    }
}
