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

import ai.djl.TrainingDivergedException;

/**
 * A {@code LearningRateTracker} tracks the evolution of the learning rate through the training
 * process.
 */
public abstract class LearningRateTracker {

    float baseLearningRate;
    int warmUpSteps;
    float warmUpBeginLearningRate;
    float warmUpFinalLearningRate;
    WarmUpMode warmUpMode;

    /**
     * A tracker returns a new learning rate based on the number of updates that have been
     * performed.
     *
     * @param builder the builder that configures learning rate options
     */
    LearningRateTracker(LrBaseBuilder<?> builder) {
        this.baseLearningRate = builder.baseLearningRate;
        this.warmUpSteps = builder.warmUpSteps;
        this.warmUpBeginLearningRate = builder.warmUpBeginLearningRate;
        this.warmUpMode = builder.warmUpMode;
        this.warmUpFinalLearningRate = baseLearningRate;
    }

    float getWarmUpLearningRate(int numUpdate) {
        float learningRate = warmUpBeginLearningRate;
        if (warmUpMode == WarmUpMode.LINEAR) {
            learningRate =
                    warmUpBeginLearningRate
                            + (warmUpFinalLearningRate - warmUpBeginLearningRate)
                                    * numUpdate
                                    / warmUpSteps;
        }
        checkLearningRate(learningRate);
        return learningRate;
    }

    /**
     * Fetches the value of the learning rate after the given number of steps/updates.
     *
     * @param numUpdate the number of steps/updates
     * @return this {@code Builder}
     */
    public abstract float getNewLearningRate(int numUpdate);

    void checkLearningRate(float learningRate) {
        if (Float.isNaN(learningRate)) {
            throw new TrainingDivergedException("Learning rate is Nan.");
        }
    }

    /**
     * Returns a new instance of {@link
     * ai.djl.training.optimizer.learningrate.FactorTracker.Builder} that can build an {@link
     * FactorTracker}.
     *
     * @return the {@link FactorTracker} {@link
     *     ai.djl.training.optimizer.learningrate.FactorTracker.Builder}
     */
    public static FactorTracker.Builder factorTracker() {
        return new FactorTracker.Builder();
    }

    /**
     * Returns a new instance of {@link
     * ai.djl.training.optimizer.learningrate.MultiFactorTracker.Builder} that can build an {@link
     * MultiFactorTracker}.
     *
     * @return the {@link MultiFactorTracker} {@link
     *     ai.djl.training.optimizer.learningrate.MultiFactorTracker.Builder}
     */
    public static MultiFactorTracker.Builder multiFactorTracker() {
        return new MultiFactorTracker.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.learningrate.MultiFactorTracker}.
     *
     * @param learningRate the fixed learning rate
     * @return the {@link MultiFactorTracker} {@link
     *     ai.djl.training.optimizer.learningrate.MultiFactorTracker.Builder}
     */
    public static FixedLearningRate fixedLearningRate(float learningRate) {
        return new FixedLearningRate.Builder().optBaseLearningRate(learningRate).build();
    }

    /** The Builder to construct a {@link LearningRateTracker}. */
    @SuppressWarnings("rawtypes")
    public abstract static class LrBaseBuilder<T extends LrBaseBuilder> {

        float baseLearningRate = 0.01f;
        int warmUpSteps;
        float warmUpBeginLearningRate;
        WarmUpMode warmUpMode = WarmUpMode.LINEAR;

        /**
         * Sets the base learning rate.
         *
         * @param baseLearningRate the base learning rate
         * @return this {@code Builder}
         */
        public T optBaseLearningRate(float baseLearningRate) {
            this.baseLearningRate = baseLearningRate;
            return self();
        }

        /**
         * Sets the number of steps until the point the learning rate is updated in warm-up mode.
         *
         * @param warmUpSteps the number of steps until the point the learning rate is updated in
         *     warm-up mode
         * @return this {@code Builder}
         */
        public T optWarmUpSteps(int warmUpSteps) {
            this.warmUpSteps = warmUpSteps;
            return self();
        }

        /**
         * Sets the value of the learning rate at the beginning of warm-up mode.
         *
         * @param warmUpBeginLearningRate the value of the learning rate at the beginning of warm-up
         *     mode
         * @return this {@code Builder}
         */
        public T optWarmUpBeginLearningRate(float warmUpBeginLearningRate) {
            this.warmUpBeginLearningRate = warmUpBeginLearningRate;
            return self();
        }

        /**
         * Sets the {@link WarmUpMode} for the {@link LearningRateTracker}.
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
