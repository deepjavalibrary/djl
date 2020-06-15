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
package ai.djl.training.optimizer.learningrate;

/** Polynomial decay learning rate. */
@SuppressWarnings("PMD")
public class PolynomialDecayLearningRateTracker extends LearningRateTracker {

    protected float endLearningRate;
    protected int decaySteps;
    protected float power;

    /**
     * Builds a PolynomialDecayLearningRateTracker.
     *
     * @param builder parameters
     */
    public PolynomialDecayLearningRateTracker(final Builder builder) {
        super(builder);
        if (Float.isNaN(builder.endLearningRate)) {
            throw new IllegalArgumentException("End learning rate is not set.");
        }
        if (builder.decaySteps <= 0) {
            throw new IllegalArgumentException("Decay steps is not set.");
        }
        this.endLearningRate = builder.endLearningRate;
        this.decaySteps = builder.decaySteps;
        this.power = builder.power;
    }

    @Override
    public float getNewLearningRate(final int numUpdate) {
        if (numUpdate < warmUpSteps) {
            return getWarmUpLearningRate(numUpdate);
        }
        int step = Math.max(0, Math.min(numUpdate - warmUpSteps, decaySteps));
        double decayedLearningRate =
                (baseLearningRate - endLearningRate)
                                * Math.pow(1.0 - (double) step / (double) decaySteps, power)
                        + endLearningRate;
        return (float) decayedLearningRate;
    }

    /**
     * Creates a new builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** Builder for PolynomialDecayLearningRateTracker. */
    public static class Builder extends LearningRateTracker.LrBaseBuilder<Builder> {

        protected float endLearningRate = Float.NaN;
        protected int decaySteps = -1;
        protected float power = 1f;

        /**
         * Sets the learning rate at which to end rate decay.
         *
         * @param endLearningRate the learning rate at which to end rate decay.
         * @return this builder
         */
        public Builder setEndLearningRate(float endLearningRate) {
            this.endLearningRate = endLearningRate;
            return self();
        }

        /**
         * Sets the number of training steps to decay learning rate in.
         *
         * @param decaySteps the number of training steps to decay learning rate in
         * @return this builder
         */
        public Builder setDecaySteps(int decaySteps) {
            this.decaySteps = decaySteps;
            return self();
        }

        /**
         * Sets the power of the polynomial to decay by.
         *
         * @param power the power of the polynomial to decay by.
         * @return this builder
         */
        public Builder optPower(float power) {
            this.power = power;
            return self();
        }

        /**
         * Builds a PolynomialDecayLearningRateTracker.
         *
         * @return a PolynomialDecayLearningRateTracker
         */
        public PolynomialDecayLearningRateTracker build() {
            return new PolynomialDecayLearningRateTracker(this);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }
}
