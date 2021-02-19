/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/**
 * Polynomial decay {@link Tracker}.
 *
 * @see Tracker
 */
public class PolynomialDecayTracker implements Tracker {

    private float baseValue;
    private float endLearningRate;
    private int decaySteps;
    private float power;

    /**
     * Builds a PolynomialDecayTracker.
     *
     * @param builder parameters
     */
    public PolynomialDecayTracker(Builder builder) {
        if (Float.isNaN(builder.endLearningRate)) {
            throw new IllegalArgumentException("End learning rate is not set.");
        }
        if (builder.decaySteps <= 0) {
            throw new IllegalArgumentException("Decay steps is not set.");
        }
        this.baseValue = builder.baseValue;
        this.endLearningRate = builder.endLearningRate;
        this.decaySteps = builder.decaySteps;
        this.power = builder.power;
    }

    /** {@inheritDoc} */
    @Override
    public float getNewValue(int numUpdate) {
        int step = Math.max(0, Math.min(numUpdate, decaySteps));
        return (float)
                ((baseValue - endLearningRate)
                                * Math.pow(1.0 - (double) step / (double) decaySteps, power)
                        + endLearningRate);
    }

    /**
     * Creates a new builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** Builder for PolynomialDecayTracker. */
    public static final class Builder {

        private float baseValue;
        private float endLearningRate = Float.NaN;
        private int decaySteps = -1;
        private float power = 1f;

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
         * Sets the learning rate at which to end rate decay.
         *
         * @param endLearningRate the learning rate at which to end rate decay.
         * @return this builder
         */
        public Builder setEndLearningRate(float endLearningRate) {
            this.endLearningRate = endLearningRate;
            return this;
        }

        /**
         * Sets the number of training steps to decay learning rate in.
         *
         * @param decaySteps the number of training steps to decay learning rate in
         * @return this builder
         */
        public Builder setDecaySteps(int decaySteps) {
            this.decaySteps = decaySteps;
            return this;
        }

        /**
         * Sets the power of the polynomial to decay by.
         *
         * @param power the power of the polynomial to decay by.
         * @return this builder
         */
        public Builder optPower(float power) {
            this.power = power;
            return this;
        }

        /**
         * Builds a PolynomialDecayTracker.
         *
         * @return a PolynomialDecayTracker
         */
        public PolynomialDecayTracker build() {
            return new PolynomialDecayTracker(this);
        }
    }
}
