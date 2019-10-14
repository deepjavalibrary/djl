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
     * @param builder Builder thant configure learning rate options
     */
    LearningRateTracker(LrBaseBuilder<?> builder) {
        this.baseLearningRate = builder.getBaseLearningRate();
        this.warmUpSteps = builder.getWarmUpSteps();
        this.warmUpBeginLearningRate = builder.getWarmUpBeginLearningRate();
        this.warmUpMode = builder.getWarmUpMode();
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

    public abstract float getNewLearningRate(int numUpdate);

    void checkLearningRate(float learningRate) {
        if (Float.isNaN(learningRate)) {
            throw new IllegalStateException("Warm up learning rate is Nan.");
        }
    }

    public static FactorTracker.Builder factorTracker() {
        return new FactorTracker.Builder();
    }

    public static MultiFactorTracker.Builder multiFactorTracker() {
        return new MultiFactorTracker.Builder();
    }

    public static FixedLearningRate fixedLearningRate(float learningRate) {
        return new FixedLearningRate.Builder().optBaseLearningRate(learningRate).build();
    }

    @SuppressWarnings("rawtypes")
    public abstract static class LrBaseBuilder<T extends LrBaseBuilder> {

        float baseLearningRate = 0.01f;
        int warmUpSteps;
        float warmUpBeginLearningRate;
        WarmUpMode warmUpMode = WarmUpMode.LINEAR;

        public T optBaseLearningRate(float baseLearningRate) {
            this.baseLearningRate = baseLearningRate;
            return self();
        }

        public T optWarmUpSteps(int warmUpSteps) {
            this.warmUpSteps = warmUpSteps;
            return self();
        }

        public T optWarmUpBeginLearningRate(float warmUpBeginLearningRate) {
            this.warmUpBeginLearningRate = warmUpBeginLearningRate;
            return self();
        }

        public T optWarmUpMode(WarmUpMode warmUpMode) {
            this.warmUpMode = warmUpMode;
            return self();
        }

        public float getBaseLearningRate() {
            return baseLearningRate;
        }

        public int getWarmUpSteps() {
            return warmUpSteps;
        }

        public float getWarmUpBeginLearningRate() {
            return warmUpBeginLearningRate;
        }

        public WarmUpMode getWarmUpMode() {
            return warmUpMode;
        }

        protected abstract T self();
    }
}
