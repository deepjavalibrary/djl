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
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Applies a single-gate recurrent layer to input. Two kinds of activation function are supported:
 * ReLU and Tanh.
 *
 * <p>Reference paper: Finding structure in time - Elman, 1988.
 * https://crl.ucsd.edu/~elman/Papers/fsit.pdf
 *
 * <p>With ReLU activation function: \(h_t = relu(W_{ih} * x_t + b_{ih} + W_{hh} * h_{(t-1)} +
 * b_{hh})\)
 *
 * <p>With Tanh activation function: \(h_t = \tanh(W_{ih} * x_t + b_{ih} + W_{hh} * h_{(t-1)} +
 * b_{hh})\)
 */
public class RNN extends RecurrentCell {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.TIME, LayoutType.BATCH, LayoutType.CHANNEL
    };

    private static final byte VERSION = 1;

    private Parameter i2hWeight;
    private Parameter h2hWeight;
    private Parameter i2hBias;
    private Parameter h2hBias;
    private Parameter state;

    /**
     * Creates a vanilla RNN block.
     *
     * @param builder the builder used to create the RNN block
     */
    RNN(Builder builder) {
        super(builder);
        mode = builder.activation == Activation.RELU ? "rnn_relu" : "rnn_tanh";
        i2hWeight = new Parameter("i2h_weight", this, ParameterType.WEIGHT);
        h2hWeight = new Parameter("h2h_weight", this, ParameterType.WEIGHT);
        i2hBias = new Parameter("i2h_bias", this, ParameterType.BIAS);
        h2hBias = new Parameter("h2h_bias", this, ParameterType.BIAS);
        state = new Parameter("state", this, ParameterType.OTHER);
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(parameterStore, inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        return ex.rnn(
                inputs,
                mode,
                stateSize,
                dropRate,
                numStackedLayers,
                useSequenceLength,
                useBidirectional,
                stateOutputs,
                params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        Shape inputShape = inputs[0];
        return new Shape[] {new Shape(inputShape.get(0), inputShape.get(1), stateSize)};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Arrays.asList(i2hWeight, i2hBias, h2hWeight, h2hBias, state);
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(Shape[] inputs) {
        this.inputShapes = inputs;
        Shape inputShape = inputs[0];
        Block.validateLayout(EXPECTED_LAYOUT, inputShape.getLayout());
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        Shape shape = inputShapes[0];
        long channelSize = shape.get(2);
        long batchSize = shape.get(1);
        switch (name) {
            case "i2h_weight":
                return new Shape(stateSize, channelSize);
            case "h2h_weight":
                return new Shape(stateSize, stateSize);
            case "i2h_bias":
            case "h2h_bias":
                return new Shape(stateSize);
            case "state":
                return new Shape(numStackedLayers, batchSize, stateSize);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        i2hWeight.save(os);
        h2hWeight.save(os);
        i2hBias.save(os);
        h2hBias.save(os);
        state.save(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        i2hWeight.load(manager, is);
        h2hWeight.load(manager, is);
        i2hBias.load(manager, is);
        h2hBias.load(manager, is);
        state.load(manager, is);
    }

    private NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        validateInputSize(inputs);
        NDArray head = inputs.head();
        Device device = head.getDevice();

        NDList result = new NDList(head);
        try (NDList parameterList = new NDList(4)) {
            parameterList.add(parameterStore.getValue(i2hWeight, device).flatten());
            parameterList.add(parameterStore.getValue(i2hBias, device).flatten());
            parameterList.add(parameterStore.getValue(h2hWeight, device).flatten());
            parameterList.add(parameterStore.getValue(h2hBias, device).flatten());

            NDArray array = NDArrays.concat(parameterList);
            result.add(array);
        }
        result.add(parameterStore.getValue(state, device));
        if (useSequenceLength) {
            result.add(inputs.get(1));
        }
        return result;
    }

    /** The Builder to construct a {@link RNN} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link RNN} block.
         *
         * @return the {@link RNN} block
         */
        public RNN build() {
            if (stateSize == -1 || numStackedLayers == -1) {
                throw new IllegalArgumentException("Must set stateSize and numStackedLayers");
            }
            return new RNN(this);
        }
    }

    /** An enum that enumerates the type of activation. */
    public enum Activation {
        RELU,
        TANH
    }
}
