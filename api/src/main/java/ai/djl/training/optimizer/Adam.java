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
package ai.djl.training.optimizer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Adam extends Optimizer {

    private LearningRateTracker learningRateTracker;
    private float beta1;
    private float beta2;
    private float epsilon;
    private boolean lazyUpdate;

    private Map<String, Map<Device, NDArray>> means;
    private Map<String, Map<Device, NDArray>> variances;

    protected Adam(Builder builder) {
        super(builder);
        learningRateTracker = builder.getLearningRateTracker();
        beta1 = builder.getBeta1();
        beta2 = builder.getBeta2();
        epsilon = builder.getEpsilon();
        lazyUpdate = builder.isLazyUpdate();
        means = new ConcurrentHashMap<>();
        variances = new ConcurrentHashMap<>();
    }

    // TODO: make this protected after integrate with PS store
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        int t = updateCount(parameterId);
        double coef1 = 1.0 - Math.pow(beta1, t);
        double coef2 = 1.0 - Math.pow(beta2, t);
        float lr = learningRateTracker.getNewLearningRate(t);
        float newLearningRate = (float) (lr * Math.sqrt(coef2) / coef1);
        float weightDecay = getWeightDecay(parameterId);
        if (Float.isNaN(newLearningRate) || Float.isNaN(weightDecay)) {
            throw new IllegalStateException("learning rate or weight decay is nan");
        }
        NDList inputs =
                new NDList(
                        weight,
                        grad,
                        withDefaultState(
                                means, parameterId, weight.getDevice(), k -> weight.zerosLike()),
                        withDefaultState(
                                variances,
                                parameterId,
                                weight.getDevice(),
                                k -> weight.zerosLike()));
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

        private LearningRateTracker learningRateTracker =
                LearningRateTracker.fixedLearningRate(0.001f);
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float epsilon = 1e-8f;
        private boolean lazyUpdate = true;

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        public Builder optLearningRateTracker(LearningRateTracker learningRateTracker) {
            this.learningRateTracker = learningRateTracker;
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

        public LearningRateTracker getLearningRateTracker() {
            return learningRateTracker;
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
