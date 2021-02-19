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
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The {@code RMSProp} {@link Optimizer}.
 *
 * <p>Two versions of RMSProp are implemented.
 *
 * <p>If `centered = False`, the algorithm described in
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by Tieleman and Hinton,
 * 2012 is used.
 *
 * <p>If `centered = True`, the algorithm described in http://arxiv.org/pdf/1308.0850v5.pdf
 * (38)-(45) by Alex Graves, 2013 is used instead.
 *
 * <p>Default version is `centered = False`.
 *
 * <p>If `centered = False`:
 *
 * <p>RMSProp updates the weights using:<br>
 * <br>
 * \( var = rho * var + (1 - rho) * grad^2 \)<br>
 * \( weight -= learning_rate * (sqrt(v) + epsilon) \)<br>
 * <br>
 * If `centered = True`: \( mean = rho * mean + (1 - rho) * grad \)<br>
 * \( var = rho * var + (1 - rho) * grad^2 \)<br>
 * \( mom = mom^2 - lr * grad / sqrt(var - mean^2) + epsilon \)<br>
 * \( weight = mean / (sqrt(var) + epsilon) \)<br>
 * <br>
 * Grad represents the gradient, mean and var are the 1st and 2nd order moment estimates (mean and
 * variance), and mom is the momentum.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/rmsprop.html">The D2L chapter on
 *     RMSProp</a>
 */
public class RmsProp extends Optimizer {

    private Tracker learningRateTracker;
    private float rho;
    private float momentum;
    private float epsilon;
    private boolean centered;

    private Map<String, Map<Device, NDArray>> means;
    private Map<String, Map<Device, NDArray>> variances;
    private Map<String, Map<Device, NDArray>> momentums;

    /**
     * Creates a new instance of {@code RMSProp} optimizer.
     *
     * @param builder the builder to create a new instance of {@code Adam} optimizer
     */
    protected RmsProp(Builder builder) {
        super(builder);
        learningRateTracker = builder.learningRateTracker;
        rho = builder.rho;
        momentum = builder.momentum;
        epsilon = builder.epsilon;
        centered = builder.centered;
        means = new ConcurrentHashMap<>();
        variances = new ConcurrentHashMap<>();
        momentums = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        float newLearningRate = learningRateTracker.getNewValue(updateCount(parameterId));
        float weightDecay = getWeightDecay();

        if (Float.isNaN(newLearningRate)
                || Float.isNaN(weightDecay)
                || Float.isInfinite(newLearningRate)
                || Float.isInfinite(weightDecay)) {
            throw new IllegalStateException("learning rate or weight decay is nan or infinite");
        }

        NDList inputs;
        if (!centered) {
            inputs =
                    new NDList(
                            weight,
                            grad,
                            withDefaultState(
                                    means,
                                    parameterId,
                                    weight.getDevice(),
                                    k -> weight.zerosLike()));
        } else {
            inputs =
                    new NDList(
                            weight,
                            grad,
                            withDefaultState(
                                    means,
                                    parameterId,
                                    weight.getDevice(),
                                    k -> weight.zerosLike()),
                            withDefaultState(
                                    variances,
                                    parameterId,
                                    weight.getDevice(),
                                    k -> weight.zerosLike()),
                            withDefaultState(
                                    momentums,
                                    parameterId,
                                    weight.getDevice(),
                                    k -> weight.zerosLike()));
        }
        NDList weights = new NDList(weight);

        float gamma1 = rho;
        float gamma2 = momentum;

        NDArrayEx ex = weight.getNDArrayInternal();

        ex.rmspropUpdate(
                inputs,
                weights,
                newLearningRate,
                weightDecay,
                rescaleGrad,
                clipGrad,
                gamma1,
                gamma2,
                epsilon,
                centered);
    }

    /**
     * Creates a builder to build a {@code RMSProp}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct an {@link RmsProp} object. */
    public static final class Builder extends OptimizerBuilder<Builder> {

        private Tracker learningRateTracker = Tracker.fixed(0.001f);
        private float rho = 0.9f;
        private float momentum = 0.9f;
        private float epsilon = 1e-8f;
        private boolean centered;

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
         * Sets the decay factor for the moving average over the past squared gradient.
         *
         * @param rho the decay factor for the moving average over past squared gradient
         * @return this {@code Builder}
         */
        public Builder optRho(float rho) {
            this.rho = rho;
            return this;
        }

        /**
         * Sets the momentum factor. This is only used if centered is set to true.
         *
         * @param momentum the momentum factor
         * @return this {@code Builder}
         */
        public Builder optMomentum(float momentum) {
            this.momentum = momentum;
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
         * Sets which version of RMSProp to use.
         *
         * <p>True: Grave's version False: Tieleman and Hinton's version
         *
         * @param centered the RMSProp version
         * @return this {@code Builder}
         */
        public Builder optCentered(boolean centered) {
            this.centered = centered;
            return this;
        }

        /**
         * Builds a {@link RmsProp} block.
         *
         * @return the {@link RmsProp} block
         */
        public RmsProp build() {
            return new RmsProp(this);
        }
    }
}
