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
import ai.djl.training.tracker.ParameterTracker;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Preconditions;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code Adam} is a generalization of the AdaGrad {@link Optimizer}.
 *
 * <p>Adam updates the weights using:<br>
 * <br>
 * \( w *= (1 - learning_rate * weight_decay\)<br>
 * \( m = beta1 * m + (1 - beta1) * grad\)<br>
 * \( v = beta2 * v + (1 - beta2) * grad^2 \)<br>
 * \( learning_rate_bias_correction = learning_rate / beta1**t * sqrt(beta2**t) \)<br>
 * \( w -= learning_rate_bias_correction * m / (sqrt(v) + epsilon) \)<br>
 * <br>
 * where g represents the gradient, and m/v are 1st and 2nd order moment estimates (mean and
 * variance), t is the step.
 *
 * @see <a href="https://pytorch.org/docs/stablew/generated/torch.optim.AdamW.html">The algorithm of
 *     AdamW</a>
 */
public class AdamW extends Optimizer {

    private ParameterTracker learningRateTracker;
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
    protected AdamW(Builder builder) {
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
        float newLearningRate = learningRateTracker.getNewValue(parameterId, t);
        float learningRateBiasCorrection = (float) (newLearningRate * Math.sqrt(coef2) / coef1);
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
                learningRateBiasCorrection,
                weightDecay,
                rescaleGrad,
                clipGrad,
                beta1,
                beta2,
                epsilon,
                true,
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

    /** The Builder to construct an {@link AdamW} object. */
    public static final class Builder extends OptimizerBuilder<Builder> {

        private ParameterTracker learningRateTracker = Tracker.fixed(0.001f);
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float epsilon = 1e-8f;

        Builder() {
            optWeightDecays(0.01f);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the {@link ParameterTracker} for this optimizer.
         *
         * @param learningRateTracker the {@link ParameterTracker} to be set
         * @return this {@code Builder}
         */
        public Builder optLearningRateTracker(ParameterTracker learningRateTracker) {
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
         * Builds a {@link AdamW} block.
         *
         * @return the {@link AdamW} block
         */
        public AdamW build() {
            return new AdamW(this);
        }
    }
}
