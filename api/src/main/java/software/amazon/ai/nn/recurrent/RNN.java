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
package software.amazon.ai.nn.recurrent;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.util.PairList;

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

    RNN(Builder builder) {
        mode = builder.getActivation() == Activation.RELU ? "rnn_relu" : "rnn_tanh";
        stateSize = builder.getStateSize();
        dropRate = builder.getDropRate();
        numStackedLayers = builder.getNumStackedLayers();
        useSequenceLength = builder.isUseSequenceLength();
        useBidirectional = builder.isUseBidirectional();
        stateOutputs = builder.isStateOutputs();
        i2hWeight = new Parameter("i2h_weight", this, ParameterType.WEIGHT);
        h2hWeight = new Parameter("h2h_weight", this, ParameterType.WEIGHT);
        i2hBias = new Parameter("i2h_bias", this, ParameterType.BIAS);
        h2hBias = new Parameter("h2h_bias", this, ParameterType.BIAS);
        state = new Parameter("state", this, ParameterType.OTHER);
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(inputs);
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
    public Shape getOutputShape(Shape... inputs) {
        Shape inputShape = inputs[0];
        return new Shape(inputShape.get(0), inputShape.get(1), stateSize);
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Arrays.asList(i2hWeight, i2hBias, h2hWeight, h2hBias, state);
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(NDList inputs) {
        NDArray input = inputs.head();
        Shape inputShape = input.getShape();
        Block.validateLayout(EXPECTED_LAYOUT, inputShape.getLayout());
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        NDArray input = inputs.get(0);
        long channelSize = input.getShape().get(2);
        long batchSize = input.getShape().get(1);
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

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        i2hWeight.save(os);
        h2hWeight.save(os);
        i2hBias.save(os);
        h2hBias.save(os);
        state.save(os);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        i2hWeight.load(manager, is);
        h2hWeight.load(manager, is);
        i2hBias.load(manager, is);
        h2hBias.load(manager, is);
        state.load(manager, is);
    }

    private NDList opInputs(NDList inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("RNN requires exactly 1 NDArray");
        }
        if (useSequenceLength && inputs.size() < 2) {
            throw new IllegalArgumentException(
                    "Input must include data, and sequenceLength as useSequenceLength is set to true");
        }

        ensureInitialized(inputs);

        NDList result = new NDList();
        NDArray parameters = i2hWeight.getArray().flatten();
        parameters =
                parameters.concat(
                        new NDArray[] {
                            i2hBias.getArray().flatten(),
                            h2hWeight.getArray().flatten(),
                            h2hBias.getArray().flatten()
                        });

        result.add(inputs.get(0));
        result.add(parameters);
        result.add(state.getArray());
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

        public RNN build() {
            if (stateSize == -1 || numStackedLayers == -1) {
                throw new IllegalArgumentException("Must set stateSize and numStackedLayers");
            }
            return new RNN(this);
        }
    }

    public enum Activation {
        RELU,
        TANH
    }
}
