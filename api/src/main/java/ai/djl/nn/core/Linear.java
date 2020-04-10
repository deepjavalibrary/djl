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
package ai.djl.nn.core;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A Linear block applies a linear transformation \(Y = XW^T + b\).
 *
 * <p>It has the following shapes:
 *
 * <p>If {@code flatten} is false:
 *
 * <ul>
 *   <li>input X: [batchSize, x1, x2, …, xn]
 *   <li>weight W: [outChannels, x1 * x2 * … * xn]
 *   <li>Bias b: [outChannels]
 *   <li>output Y: [batchSize, outChannels]
 * </ul>
 *
 * <p>If {@code flatten} is false:
 *
 * <ul>
 *   <li>input X: [x1, x2, …, xn, input_dim]
 *   <li>weight W: [outChannels, input_dim]
 *   <li>Bias b: [outChannels]
 *   <li>output Y: [x1, x2, …, xn, outChannels]
 * </ul>
 *
 * <p>The Linear block should be constructed using {@link Linear.Builder}.
 */
public class Linear extends ParameterBlock {

    private static final byte VERSION = 2;

    private long outChannels;
    private long inputDimension;
    private boolean flatten;

    private Shape inputShape;

    private Parameter weight;
    private Parameter bias;

    Linear(Builder builder) {
        outChannels = builder.outChannels;
        flatten = builder.flatten;
        weight = new Parameter("weight", this, ParameterType.WEIGHT);
        if (builder.bias) {
            bias = new Parameter("bias", this, ParameterType.BIAS);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(parameterStore, inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        return ex.fullyConnected(inputs, outChannels, flatten, bias == null, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        if (flatten) {
            return new Shape[] {new Shape(inputs[0].get(0), outChannels)};
        }
        return new Shape[] {inputShape.addAll(new Shape(outChannels))};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        if (bias != null) {
            return Arrays.asList(weight, bias);
        }
        return Collections.singletonList(weight);
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        return new PairList<>(
                Collections.singletonList("linearInput"), Collections.singletonList(inputShape));
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(Shape[] inputShapes) {
        this.inputShapes = inputShapes;
        Shape input = inputShapes[0];
        if (flatten) {
            Shape inChannels;
            if (input.isLayoutKnown()) {
                inChannels = input.filterByLayoutType(t -> !t.equals(LayoutType.BATCH));
                inputShape =
                        input.map(
                                pair ->
                                        new Pair<>(
                                                pair.getValue().equals(LayoutType.BATCH)
                                                        ? Long.valueOf(-1)
                                                        : pair.getKey(),
                                                pair.getValue()));
            } else if (input.dimension() > 1) {
                inChannels = input.slice(1);
                inputShape =
                        new Shape(new long[] {-1}, new LayoutType[] {LayoutType.BATCH})
                                .addAll(input.slice(1));
            } else {
                inChannels = input;
                inputShape = input;
            }
            inputDimension = inChannels.size();
        } else {
            inputDimension = input.get(input.dimension() - 1);
            inputShape = input.slice(0, input.dimension() - 1);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        switch (name) {
            case "weight":
                return new Shape(outChannels, inputDimension);
            case "bias":
                return new Shape(outChannels);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        os.writeBoolean(flatten);
        os.writeLong(inputDimension);
        os.write(inputShape.getEncoded());
        weight.save(os);
        if (bias != null) {
            bias.save(os);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version == VERSION) {
            flatten = is.readBoolean();
            inputDimension = is.readLong();
        } else if (version == 1) {
            flatten = false;
            inputDimension = Shape.decode(is).size();
        } else {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        inputShape = Shape.decode(is);
        weight.load(manager, is);
        if (bias != null) {
            bias.load(manager, is);
        }
    }

    private NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Linear requires exactly 1 NDArray");
        }
        Device device = inputs.head().getDevice();

        NDList result = new NDList(inputs);
        result.add(parameterStore.getValue(weight, device));
        if (bias != null) {
            result.add(parameterStore.getValue(bias, device));
        }
        return result;
    }

    /**
     * Creates a builder to build a {@code Linear}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Linear} type of {@link Block}. */
    public static final class Builder {

        private long outChannels;
        private boolean bias = true;
        private boolean flatten;

        Builder() {}

        /**
         * Sets the number of output channels.
         *
         * @param outChannels the number of desired output channels
         * @return this Builder
         */
        public Builder setOutChannels(long outChannels) {
            this.outChannels = outChannels;
            return this;
        }

        /**
         * Sets the optional parameter that indicates whether to include a bias vector with default
         * value of true.
         *
         * @param bias whether to use a bias vector parameter
         * @return this Builder
         */
        public Builder optBias(boolean bias) {
            this.bias = bias;
            return this;
        }

        /**
         * Sets the optional parameter that indicates whether the input tensor should be flattened.
         *
         * <p>If flatten is set to true, all but the first axis of input data are collapsed
         * together. If false, all but the last axis of input data are kept the same, and the
         * transformation applies on the last axis.
         *
         * @param flatten whether the input tensor should be flattened.
         * @return this Builder
         */
        public Builder optFlatten(boolean flatten) {
            this.flatten = flatten;
            return this;
        }

        /**
         * Returns the constructed {@code Linear}.
         *
         * @return the constructed {@code Linear}
         * @throws IllegalArgumentException if all required parameters (outChannels) have not been
         *     set
         */
        public Linear build() {
            if (outChannels == 0) {
                throw new IllegalArgumentException("You must specify outChannels");
            }
            return new Linear(this);
        }
    }
}
