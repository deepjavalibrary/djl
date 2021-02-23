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
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code Adadelta} is an Adadelta {@code Optimizer}.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/adadelta.html">The D2L chapter on
 *     Adadelta</a>
 */
public class Adadelta extends Optimizer {

    private float rho;
    private float epsilon;
    private Map<String, Map<Device, NDArray>> accumG;
    private Map<String, Map<Device, NDArray>> accumDelta;

    /**
     * Creates a new instance of {@code Adadelta}.
     *
     * @param builder the builder to create a new instance of {@link Adadelta}
     */
    protected Adadelta(Builder builder) {
        super(builder);
        rho = builder.rho;
        epsilon = builder.epsilon;
        accumG = new ConcurrentHashMap<>();
        accumDelta = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        float weightDecay = getWeightDecay();
        NDList inputs =
                new NDList(
                        weight,
                        grad,
                        withDefaultState(
                                accumG, parameterId, weight.getDevice(), k -> weight.zerosLike()),
                        withDefaultState(
                                accumDelta,
                                parameterId,
                                weight.getDevice(),
                                k -> weight.zerosLike()));

        NDList weights = new NDList(weight);

        NDArrayEx ex = weight.getNDArrayInternal();
        ex.adadeltaUpdate(inputs, weights, weightDecay, rescaleGrad, clipGrad, rho, epsilon);
    }

    /** The Builder to construct an {@link Adadelta} object. */
    public static final class Builder extends OptimizerBuilder<Builder> {

        private float rho = 0.9f;
        private float epsilon = 1e-8f;

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the rho for {@link Adadelta}.
         *
         * @param rho the value of rho
         * @return this {@code Builder}
         */
        public Builder optRho(float rho) {
            this.rho = rho;
            return this;
        }

        /**
         * Sets \(epsilon\) - a small quantity for numerical stability.
         *
         * @param epsilon a small quantity for numerical stability
         * @return this {@code Builder}
         */
        public Adadelta.Builder optEpsilon(float epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        /**
         * Builds a {@link Adadelta} block.
         *
         * @return the {@link Adadelta} block
         */
        public Adadelta build() {
            return new Adadelta(this);
        }
    }
}
