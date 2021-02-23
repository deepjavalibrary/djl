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
 * {@code CosineTracker} is an implementation of {@link Tracker} which is updated by taking sections
 * of a cosine curve to smoothly reduce learning rate until a specified step and base learning rate.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/lr-scheduler.html#cosine-tracker">For
 *     tracking learning rates, this section in the D2L chapter on learning rate scheduling</a>
 */
public class CosineTracker implements Tracker {

    private float baseValue;
    private float finalValue;
    private int maxUpdates;

    /**
     * Creates a new instance of {@code CosineTracker}.
     *
     * @param builder the builder to create a new instance of {@code CosineTracker}
     */
    public CosineTracker(Builder builder) {
        this.baseValue = builder.baseValue;
        this.finalValue = builder.finalValue;
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
            return finalValue;
        }
        // Scale the curve to fit the number of steps
        float step =
                (baseValue - finalValue)
                        / 2
                        * (1 + (float) Math.cos(Math.PI * numUpdate / maxUpdates));
        return finalValue + step;
    }

    /** The Builder to construct an {@link CosineTracker} object. */
    public static final class Builder {

        private float baseValue;
        private float finalValue = 0.01f;
        int maxUpdates;

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
         * Sets the final value that the learning rate will remain constant as after the specified
         * max number of updates.
         *
         * @param finalValue the final value
         * @return this {@code Builder}
         */
        public Builder optFinalValue(float finalValue) {
            this.finalValue = finalValue;
            return this;
        }

        /**
         * Sets the maximum number of updates after which the value should remain constant.
         *
         * @param maxUpdates the maximum number of updates after which the value should remain
         *     constant
         * @return this {@code Builder}
         */
        public Builder setMaxUpdates(int maxUpdates) {
            this.maxUpdates = maxUpdates;
            return this;
        }

        /**
         * Builds a {@link CosineTracker} block.
         *
         * @return the {@link CosineTracker} block
         */
        public CosineTracker build() {
            Preconditions.checkArgument(baseValue > 0, "You must set a starting learning rate!");
            Preconditions.checkArgument(
                    maxUpdates > 0, "You must set a maximum number of updates!");
            Preconditions.checkArgument(
                    baseValue > finalValue,
                    "Starting learning rate must be greater than final learning rate!");
            return new CosineTracker(this);
        }
    }
}
