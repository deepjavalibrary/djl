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
 * <ul>
 *   <li>input X: [batchSize..., inChannels]
 *   <li>weight W: [outChannels, inChannels]
 *   <li>Bias b: [outChannels]
 *   <li>output Y: [batchSize..., outChannels]
 * </ul>
 *
 * <p>The Linear block should be constructed using {@link Linear.Builder}.
 */
public class Linear extends ParameterBlock {

    private static final byte VERSION = 1;

    private long outChannels;

    private Shape inputShape;
    private Shape inChannels;

    private Parameter weight;
    private Parameter bias;

    Linear(Builder builder) {
        outChannels = builder.outChannels;
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
        return ex.fullyConnected(inputs, outChannels, false, bias == null, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        return new Shape[] {new Shape(inputs[0].get(0), outChannels)};
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
            inChannels = input.slice(0);
            inputShape = input;
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        switch (name) {
            case "weight":
                return new Shape(outChannels).addAll(inChannels);
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
        os.write(inChannels.getEncoded());
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
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        inChannels = Shape.decode(is);
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

    /** The Builder to construct a {@link Linear} type of {@link Block}. */
    public static final class Builder {

        private long outChannels;
        private boolean bias = true;

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
