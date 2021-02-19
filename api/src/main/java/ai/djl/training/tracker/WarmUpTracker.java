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

import ai.djl.TrainingDivergedException;

/**
 * A {@code WarmUpTracker} applies a simple warm-up before executing a main {@link Tracker}.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/lr-scheduler.html#warmup">For tracking
 *     learning rates, this section in the D2L chapter on learning rate scheduling</a>
 */
public final class WarmUpTracker implements Tracker {

    Tracker mainTracker;
    int warmUpSteps;
    float warmUpBeginValue;
    float warmUpFinalValue;
    Mode warmUpMode;

    /**
     * A tracker returns a new value based on the number of updates that have been performed.
     *
     * @param builder the builder that configures tracker options
     */
    WarmUpTracker(Builder builder) {
        this.mainTracker = builder.mainTracker;
        this.warmUpSteps = builder.warmUpSteps;
        this.warmUpBeginValue = builder.warmUpBeginValue;
        this.warmUpMode = builder.warmUpMode;
        this.warmUpFinalValue = mainTracker.getNewValue(0);
    }

    /**
     * Creates a new builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    float getWarmUpValue(int numUpdate) {
        float value = warmUpBeginValue;
        if (warmUpMode == Mode.LINEAR) {
            value =
                    warmUpBeginValue
                            + (warmUpFinalValue - warmUpBeginValue) * numUpdate / warmUpSteps;
        }
        checkValue(value);
        return value;
    }

    /** {@inheritDoc} */
    @Override
    public float getNewValue(int numUpdate) {
        if (numUpdate < warmUpSteps) {
            return getWarmUpValue(numUpdate);
        } else {
            return mainTracker.getNewValue(numUpdate - warmUpSteps);
        }
    }

    void checkValue(float value) {
        if (Float.isNaN(value)) {
            throw new TrainingDivergedException("Value is Nan.");
        }
    }

    /** The Builder to construct a {@link WarmUpTracker}. */
    @SuppressWarnings("rawtypes")
    public static final class Builder {

        Tracker mainTracker;
        int warmUpSteps;
        float warmUpBeginValue;
        Mode warmUpMode = Mode.LINEAR;

        private Builder() {}

        /**
         * Sets the base value.
         *
         * @param mainTracker the tracker to use after warm up ends
         * @return this {@code Builder}
         */
        public Builder setMainTracker(Tracker mainTracker) {
            this.mainTracker = mainTracker;
            return this;
        }

        /**
         * Sets the number of steps until the point the value is updated in warm-up mode.
         *
         * @param warmUpSteps the number of steps the value is updated in warm-up mode
         * @return this {@code Builder}
         */
        public Builder optWarmUpSteps(int warmUpSteps) {
            this.warmUpSteps = warmUpSteps;
            return this;
        }

        /**
         * Sets the value at the beginning of warm-up mode.
         *
         * @param warmUpBeginValue the value at the beginning of warm-up mode
         * @return this {@code Builder}
         */
        public Builder optWarmUpBeginValue(float warmUpBeginValue) {
            this.warmUpBeginValue = warmUpBeginValue;
            return this;
        }

        /**
         * Sets the {@link Mode} for the {@link WarmUpTracker}.
         *
         * @param warmUpMode the {@link Mode} to be set
         * @return this {@code Builder}
         */
        public Builder optWarmUpMode(Mode warmUpMode) {
            this.warmUpMode = warmUpMode;
            return this;
        }

        /**
         * Builds a {@link WarmUpTracker} block.
         *
         * @return the {@link WarmUpTracker} block
         */
        public WarmUpTracker build() {
            return new WarmUpTracker(this);
        }
    }

    /** An enum that enumerates the types of warm-up modes for a {@link WarmUpTracker}. */
    public enum Mode {
        LINEAR,
        CONSTANT
    }
}
