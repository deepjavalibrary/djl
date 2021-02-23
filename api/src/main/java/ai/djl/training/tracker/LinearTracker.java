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

import ai.djl.training.tracker.WarmUpTracker.Builder;
import ai.djl.util.Preconditions;

/**
 * {@code FactorTracker} is an implementation of {@link Tracker} which is updated by a constant
 * factor.
 *
 * @see Tracker
 */
public class LinearTracker implements Tracker {

    private float baseValue;
    private float slope;
    private int maxUpdates;

    /**
     * Creates a new instance of {@code FactorTracker}.
     *
     * @param builder the builder to create a new instance of {@code FactorTracker}
     */
    public LinearTracker(Builder builder) {
        this.baseValue = builder.baseValue;
        this.slope = builder.slope;
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
        return baseValue + numUpdate * slope;
    }

    /** The Builder to construct an {@link LinearTracker} object. */
    public static final class Builder {

        float baseValue;
        float slope;
        Float min;
        Float max;
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
         * Sets the value of the linear slope.
         *
         * <p>Use a positive number for an increasing value and negative for decreasing.
         *
         * @param slope the value of the linear slope
         * @return this {@code Builder}
         */
        public Builder optSlope(float slope) {
            this.slope = slope;
            return this;
        }

        /**
         * Sets the maximum value for a positive slope.
         *
         * <p>This is equivalent to the max updates. Only one can be set.
         *
         * @param max the max value
         * @return this {@code Builder}
         */
        public Builder optMaxValue(float max) {
            this.max = max;
            return this;
        }

        /**
         * Sets the minimum value for a negative slope.
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
         * Builds a {@link LinearTracker} block.
         *
         * @return the {@link LinearTracker} block
         */
        public LinearTracker build() {
            Preconditions.checkArgument(slope != 0, "You must set a slope");
            Preconditions.checkArgument(
                    min == null || max == null, "You can not set both a max value and a min value");

            if (max != null) {
                Preconditions.checkArgument(
                        maxUpdates == null, "You can not set both maxUpdates and a max value");
                Preconditions.checkArgument(
                        slope > 0, "The slope must be positive for a max value");
                Preconditions.checkArgument(
                        max > baseValue, "The max must be greater than the base value");
                maxUpdates = Math.round((max - baseValue) / slope);
            } else if (min != null) {
                Preconditions.checkArgument(
                        maxUpdates == null, "You can not set both maxUpdates and a min value");
                Preconditions.checkArgument(
                        slope < 0, "The slope must be negative for a min value");
                Preconditions.checkArgument(
                        min < baseValue, "The min must be smaller than the base value");
                maxUpdates = -Math.round((baseValue - min) / slope);
            } else if (maxUpdates == null) {
                // Default to no max if none set
                maxUpdates = Integer.MAX_VALUE;
            }

            return new LinearTracker(this);
        }
    }
}
