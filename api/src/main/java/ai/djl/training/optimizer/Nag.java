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
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code Nag} is a Nesterov accelerated gradient optimizer.
 *
 * <p>This optimizer updates each weight by:<br>
 * \( state = momentum * state + grad + wd *weight\)<br>
 * \( weight = weight - (lr * (grad + momentum * state))<br>
 */
public class Nag extends Optimizer {

    private Tracker learningRateTracker;
    private float momentum;
    private Map<String, Map<Device, NDArray>> momentumStates;

    /**
     * Creates a new instance of {@code Nag} optimizer.
     *
     * @param builder the builder to create a new instance of {@code Nag} optimizer
     */
    protected Nag(Builder builder) {
        super(builder);
        learningRateTracker = builder.learningRateTracker;
        momentum = builder.momentum;
        momentumStates = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        // TODO: Support Mixed precision Sparse
        float newLearningRate = learningRateTracker.getNewValue(updateCount(parameterId));
        float weightDecay = getWeightDecay();
        NDList inputs;
        if (momentum != 0f) {
            inputs =
                    new NDList(
                            weight,
                            grad,
                            withDefaultState(
                                    momentumStates,
                                    parameterId,
                                    weight.getDevice(),
                                    k -> weight.zerosLike()));
        } else {
            inputs = new NDList(weight, grad);
        }
        NDList weights = new NDList(weight);

        NDArrayEx ex = weight.getNDArrayInternal();
        ex.nagUpdate(
                inputs, weights, newLearningRate, weightDecay, rescaleGrad, clipGrad, momentum);
    }

    /** The Builder to construct an {@link Nag} object. */
    public static final class Builder extends OptimizerBuilder<Builder> {

        Tracker learningRateTracker;
        float momentum;

        Builder() {}

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
         * Sets the momentum for {@link Nag}.
         *
         * @param momentum the value of momentum
         * @return this {@code Builder}
         */
        public Builder setMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Nag} block.
         *
         * @return the {@link Nag} block
         */
        public Nag build() {
            Objects.requireNonNull(learningRateTracker, "No lrTracker set");
            Preconditions.checkArgument(momentum != 0, "The momentum should be set");
            return new Nag(this);
        }
    }
}
