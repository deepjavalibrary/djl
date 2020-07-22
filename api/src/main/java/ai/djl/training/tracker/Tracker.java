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
 * A {@code Tracker} represents a hyper-parameter that changes gradually through the training
 * process.
 */
public abstract class Tracker {

    float baseValue;
    int warmUpSteps;
    float warmUpBeginValue;
    float warmUpFinalValue;
    WarmUpMode warmUpMode;

    /**
     * A tracker returns a new value based on the number of updates that have been performed.
     *
     * @param builder the builder that configures tracker options
     */
    Tracker(TrackerBaseBuilder<?> builder) {
        this.baseValue = builder.baseValue;
        this.warmUpSteps = builder.warmUpSteps;
        this.warmUpBeginValue = builder.warmUpBeginValue;
        this.warmUpMode = builder.warmUpMode;
        this.warmUpFinalValue = baseValue;
    }

    float getWarmUpValue(int numUpdate) {
        float value = warmUpBeginValue;
        if (warmUpMode == WarmUpMode.LINEAR) {
            value =
                    warmUpBeginValue
                            + (warmUpFinalValue - warmUpBeginValue) * numUpdate / warmUpSteps;
        }
        checkValue(value);
        return value;
    }

    /**
     * Fetches the value after the given number of steps/updates.
     *
     * @param numUpdate the number of steps/updates
     * @return this {@code Builder}
     */
    public abstract float getNewValue(int numUpdate);

    void checkValue(float value) {
        if (Float.isNaN(value)) {
            throw new TrainingDivergedException("Value is Nan.");
        }
    }

    /**
     * Returns a new instance of {@link ai.djl.training.tracker.FactorTracker.Builder} that can
     * build an {@link FactorTracker}.
     *
     * @return the {@link FactorTracker} {@link ai.djl.training.tracker.FactorTracker.Builder}
     */
    public static FactorTracker.Builder factorTracker() {
        return new FactorTracker.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.tracker.MultiFactorTracker.Builder} that can
     * build an {@link MultiFactorTracker}.
     *
     * @return the {@link MultiFactorTracker} {@link
     *     ai.djl.training.tracker.MultiFactorTracker.Builder}
     */
    public static MultiFactorTracker.Builder multiFactorTracker() {
        return new MultiFactorTracker.Builder();
    }

    /**
     * Returns a new instance of {@link Tracker} with a fixed value.
     *
     * @param value the fixed value
     * @return a instance of {@link Tracker} with a fixed value
     */
    public static Tracker fixed(float value) {
        return FixedTracker.builder().optBaseValue(value).build();
    }

    /** The Builder to construct a {@link Tracker}. */
    @SuppressWarnings("rawtypes")
    public abstract static class TrackerBaseBuilder<T extends TrackerBaseBuilder> {

        float baseValue = 0.01f;
        int warmUpSteps;
        float warmUpBeginValue;
        WarmUpMode warmUpMode = WarmUpMode.LINEAR;

        /**
         * Sets the base value.
         *
         * @param baseValue the base value
         * @return this {@code Builder}
         */
        public T optBaseValue(float baseValue) {
            this.baseValue = baseValue;
            return self();
        }

        /**
         * Sets the number of steps until the point the value is updated in warm-up mode.
         *
         * @param warmUpSteps the number of steps the value is updated in warm-up mode
         * @return this {@code Builder}
         */
        public T optWarmUpSteps(int warmUpSteps) {
            this.warmUpSteps = warmUpSteps;
            return self();
        }

        /**
         * Sets the value at the beginning of warm-up mode.
         *
         * @param warmUpBeginValue the value at the beginning of warm-up mode
         * @return this {@code Builder}
         */
        public T optWarmUpBeginValue(float warmUpBeginValue) {
            this.warmUpBeginValue = warmUpBeginValue;
            return self();
        }

        /**
         * Sets the {@link WarmUpMode} for the {@link Tracker}.
         *
         * @param warmUpMode the {@link WarmUpMode} to be set
         * @return this {@code Builder}
         */
        public T optWarmUpMode(WarmUpMode warmUpMode) {
            this.warmUpMode = warmUpMode;
            return self();
        }

        protected abstract T self();
    }
}
