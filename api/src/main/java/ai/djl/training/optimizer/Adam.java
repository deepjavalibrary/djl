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
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Preconditions;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code Adam} is a generalization of the AdaGrad {@link Optimizer}.
 *
 * <p>Adam updates the weights using:<br>
 * <br>
 * \( m = beta1 * m + (1 - beta1) * grad\)<br>
 * \( v = beta2 * v + (1 - beta2) * grad^2 \)<br>
 * \( w -= learning_rate * m / (sqrt(v) + epsilon) \)<br>
 * <br>
 * where g represents the gradient, and m/v are 1st and 2nd order moment estimates (mean and
 * variance).
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/adam.html">The D2L chapter on Adam</a>
 */
public class Adam extends Optimizer {

    private Tracker learningRateTracker;
    private float beta1;
    private float beta2;
    private float epsilon;

    private Map<String, Map<Device, NDArray>> means;
    private Map<String, Map<Device, NDArray>> variances;

    /**
     * Creates a new instance of {@code Adam} optimizer.
     *
     * @param builder the builder to create a new instance of {@code Adam} optimizer
     */
    protected Adam(Builder builder) {
        super(builder);
        learningRateTracker = builder.learningRateTracker;
        beta1 = builder.beta1;
        beta2 = builder.beta2;
        epsilon = builder.epsilon;
        means = new ConcurrentHashMap<>();
        variances = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        int t = updateCount(parameterId);
        double coef1 = 1.0 - Math.pow(beta1, t);
        double coef2 = 1.0 - Math.pow(beta2, t);
        float lr = learningRateTracker.getNewValue(t);
        float newLearningRate = (float) (lr * Math.sqrt(coef2) / coef1);
        float weightDecay = getWeightDecay();

        Preconditions.checkArgument(
                !Float.isNaN(newLearningRate)
                        && !Float.isNaN(weightDecay)
                        && !Float.isInfinite(newLearningRate)
                        && !Float.isInfinite(weightDecay),
                "learning rate or weight decay is nan or infinite");
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
                true);
    }

    /**
     * Creates a builder to build a {@code Adam}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct an {@link Adam} object. */
    public static final class Builder extends OptimizerBuilder<Builder> {

        private Tracker learningRateTracker = Tracker.fixed(0.001f);
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float epsilon = 1e-8f;

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the {@link Tracker} for this optimizer.
         *
         * @param learningRateTracker the {@link Tracker} to be set
         * @return this {@code Builder}
         */
        public Builder optLearningRateTracker(Tracker learningRateTracker) {
            this.learningRateTracker = learningRateTracker;
            return this;
        }

        /**
         * Sets the decay rate for the first moment estimates.
         *
         * @param beta1 the deacay rate for the the first moment estimates
         * @return this {@code Builder}
         */
        public Builder optBeta1(float beta1) {
            this.beta1 = beta1;
            return this;
        }

        /**
         * Sets the decay rate for the second moment estimates.
         *
         * @param beta2 the decay rate for the the second moment estimates
         * @return this {@code Builder}
         */
        public Builder optBeta2(float beta2) {
            this.beta2 = beta2;
            return this;
        }

        /**
         * Sets \(epsilon\) - a small quantity for numerical stability.
         *
         * @param epsilon a small quantity for numerical stability
         * @return this {@code Builder}
         */
        public Builder optEpsilon(float epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        /**
         * Builds a {@link Adam} block.
         *
         * @return the {@link Adam} block
         */
        public Adam build() {
            return new Adam(this);
        }
    }
}
