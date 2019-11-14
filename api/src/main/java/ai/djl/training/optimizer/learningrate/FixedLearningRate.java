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

/**
 * {@code FixedLearningRate} is an implementation of {@link LearningRateTracker} which returns a
 * fixed value for the learning rate.
 */
class FixedLearningRate extends LearningRateTracker {

    /**
     * Creates a new instance of {@code FixedLearningRate}.
     *
     * @param builder the builder used to build this object
     */
    public FixedLearningRate(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public float getNewLearningRate(int numUpdate) {
        return baseLearningRate;
    }
    /** The Builder to construct an {@link FixedLearningRate} object. */
    public static final class Builder extends LrBaseBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link FixedLearningRate} block.
         *
         * @return the {@link FixedLearningRate} block
         */
        public FixedLearningRate build() {
            return new FixedLearningRate(this);
        }
    }
}
