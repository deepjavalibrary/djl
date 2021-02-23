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
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code Sgd} is a Stochastic Gradient Descent (SGD) optimizer.
 *
 * <p>If momentum is not set, it updates weights using the following update function:<br>
 * \( weight = weight - learning_rate * (gradient + wd * weight) \).
 *
 * <p>If momentum is set, it updates weights using the following update function:<br>
 * \( state = momentum * state + learning_rate * gradient \)<br>
 * \( weight -= state \)<br>
 * Momentum update has better convergence rates on neural networks.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/sgd.html">The D2L chapter on SGD</a>
 */
public class Sgd extends Optimizer {

    private Tracker learningRateTracker;
    private float momentum;
    private Map<String, Map<Device, NDArray>> momentumStates;

    /**
     * Creates a new instance of {@code Sgd}.
     *
     * @param builder the builder to create a new instance of {@link Sgd}
     */
    protected Sgd(Builder builder) {
        super(builder);
        learningRateTracker = builder.learningRateTracker;
        momentum = builder.momentum;
        momentumStates = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        // TODO: Support Mixed precision Sparse
        float weightDecay = getWeightDecay();
        float learningRate = learningRateTracker.getNewValue(updateCount(parameterId));
        NDList inputs;
        if (momentum != 0f) {
            NDArray state =
                    withDefaultState(
                            momentumStates,
                            parameterId,
                            weight.getDevice(),
                            k -> weight.zerosLike());
            inputs = new NDList(weight, grad, state);
        } else {
            inputs = new NDList(weight, grad);
        }
        NDList weights = new NDList(weight);

        NDArrayEx ex = weight.getNDArrayInternal();
        ex.sgdUpdate(
                inputs, weights, learningRate, weightDecay, rescaleGrad, clipGrad, momentum, true);
    }

    /** The Builder to construct an {@link Sgd} object. */
    public static final class Builder extends OptimizerBuilder<Builder> {

        Tracker learningRateTracker;
        float momentum;

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
        public Builder setLearningRateTracker(Tracker learningRateTracker) {
            this.learningRateTracker = learningRateTracker;
            return this;
        }

        /**
         * Sets the momentum for {@link Sgd}.
         *
         * @param momentum the value of momentum
         * @return this {@code Builder}
         */
        public Builder optMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        /**
         * Builds a {@link Sgd} block.
         *
         * @return the {@link Sgd} block
         */
        public Sgd build() {
            Objects.requireNonNull(learningRateTracker, "No lrTracker set");
            return new Sgd(this);
        }
    }
}
