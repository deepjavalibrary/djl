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

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.util.PairList;

public class Adam extends Optimizer {

    private float learningRate;
    private float beta1;
    private float beta2;
    private float epsilon;
    private boolean lazyUpdate;

    private NDList means;
    private NDList variances;

    protected Adam(Builder builder) {
        super(builder);
        learningRate = builder.getLearningRate();
        beta1 = builder.getBeta1();
        beta2 = builder.getBeta2();
        epsilon = builder.getEpsilon();
        lazyUpdate = builder.isLazyUpdate();
    }

    @Override
    protected boolean initializeStates(PairList<String, Parameter> parameters) {
        if (means == null) {
            means = new NDList(parameters.size());
            variances = new NDList(parameters.size());
            for (Parameter param : parameters.values()) {
                means.add(param.getArray().zerosLike());
                variances.add(param.getArray().zerosLike());
            }
        }
        return true;
    }

    @Override
    protected void update(int index, NDArray weight, NDArray grad) {
        double t = updateCount(index);
        double coef1 = 1.0 - Math.pow(beta1, t);
        double coef2 = 1.0 - Math.pow(beta2, t);
        float newLearningRate = (float) (learningRate * Math.sqrt(coef2) / coef1);

        float weightDecay = getWeightDecay(index);
        NDList inputs = new NDList(weight, grad, means.get(index), variances.get(index));
        NDList weights = new NDList(weight);

        NDArrayEx ex = weight.getNDArrayInternal();

        ex.adamUpdate(
                inputs,
                weights,
                newLearningRate,
                weightDecay,
                rescaleGrad,
                clipGrad,
                beta1,
                beta2,
                epsilon,
                lazyUpdate);
    }

    public static final class Builder extends BaseBuilder<Builder> {

        private float learningRate = 0.001f;
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float epsilon = 1e-8f;
        private boolean lazyUpdate = true;

        @Override
        protected Builder self() {
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
            return new Adam(this);
        }
    }
}
