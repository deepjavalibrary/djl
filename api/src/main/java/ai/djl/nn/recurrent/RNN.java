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
package ai.djl.nn.recurrent;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;

/**
 * {@code RNN} is an implementation of recurrent neural networks which applies a single-gate
 * recurrent layer to input. Two kinds of activation function are supported: ReLU and Tanh.
 *
 * <p>Current implementation refers the [paper](https://crl.ucsd.edu/~elman/Papers/fsit.pdf),
 * Finding structure in time - Elman, 1988.
 *
 * <p>The RNN operator is formulated as below:
 *
 * <p>With ReLU activation function: \(h_t = relu(W_{ih} * x_t + b_{ih} + W_{hh} * h_{(t-1)} +
 * b_{hh})\)
 *
 * <p>With Tanh activation function: \(h_t = \tanh(W_{ih} * x_t + b_{ih} + W_{hh} * h_{(t-1)} +
 * b_{hh})\)
 */
public class RNN extends RecurrentBlock {

    private Activation activation;

    /**
     * Creates a vanilla RNN block.
     *
     * @param builder the builder used to create the RNN block
     */
    RNN(Builder builder) {
        super(builder);
        activation = builder.activation;
        gates = 1;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        Device device = inputs.head().getDevice();
        NDList rnnParams = new NDList();
        for (Parameter parameter : parameters.values()) {
            rnnParams.add(parameterStore.getValue(parameter, device, training));
        }

        NDArray input = inputs.head();
        if (inputs.size() == 1) {
            int batchIndex = batchFirst ? 0 : 1;
            inputs.add(
                    input.getManager()
                            .zeros(
                                    new Shape(
                                            (long) numLayers * getNumDirections(),
                                            input.size(batchIndex),
                                            stateSize)));
        }
        NDList outputs =
                ex.rnn(
                        input,
                        inputs.get(1),
                        rnnParams,
                        hasBiases,
                        numLayers,
                        activation,
                        dropRate,
                        training,
                        bidirectional,
                        batchFirst);
        if (returnState) {
            return outputs;
        }
        outputs.stream().skip(1).forEach(NDArray::close);
        return new NDList(outputs.get(0));
    }

    /**
     * Creates a builder to build a {@link RNN}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link RNN} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the activation for the RNN - ReLu or Tanh.
         *
         * @param activation the activation
         * @return this Builder
         */
        public Builder setActivation(RNN.Activation activation) {
            this.activation = activation;
            return self();
        }

        /**
         * Builds a {@link RNN} block.
         *
         * @return the {@link RNN} block
         */
        public RNN build() {
            Preconditions.checkArgument(
                    stateSize > 0 && numLayers > 0, "Must set stateSize and numLayers");
            return new RNN(this);
        }
    }

    /** An enum that enumerates the type of activation. */
    public enum Activation {
        RELU,
        TANH
    }
}
