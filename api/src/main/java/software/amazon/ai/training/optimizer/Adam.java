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
package software.amazon.ai.training.optimizer;

public interface Adam extends Optimizer {

    class Builder extends BaseBuilder<Builder> {

        private float learningRate = 0.001f;
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float epsilon = 1e-8f;
        private boolean lazyUpdate = true;

        @Override
        Builder self() {
            return this;
        }

        public Builder optLearningRate(float learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder optBeta1(float beta1) {
            this.beta1 = beta1;
            return this;
        }

        public Builder optBeta2(float beta2) {
            this.beta2 = beta2;
            return this;
        }

        public Builder optEpsilon(float epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public Builder optLazyUpdate(boolean lazyUpdate) {
            this.lazyUpdate = lazyUpdate;
            return this;
        }

        public float getLearningRate() {
            return learningRate;
        }

        public float getBeta1() {
            return beta1;
        }

        public float getBeta2() {
            return beta2;
        }

        public float getEpsilon() {
            return epsilon;
        }

        public boolean isLazyUpdate() {
            return lazyUpdate;
        }

        public Adam build() {
            return factory.createAdam(this);
        }
    }
}
