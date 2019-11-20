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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Applies GRU (Gated Recurrent Unit) recurrent layer to input.
 *
 * <p>Reference paper - Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078. The
 * definition of GRU here is slightly different from the paper but compatible with CUDNN.
 *
 * <p>$$ \begin{split}\begin{array}{ll} r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr}
 * h_{(t-1)} + b_{hr}) \\ z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
 * n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\ h_t = (1 - z_t) * n_t +
 * z_t * h_{(t-1)} \\ \end{array}\end{split} $$
 */
public class GRU extends RecurrentCell {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.TIME, LayoutType.BATCH, LayoutType.CHANNEL
    };

    private static final byte VERSION = 1;

    private List<Parameter> parameters =
            Arrays.asList(
                    new Parameter("i2rWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2rBias", this, ParameterType.BIAS),
                    new Parameter("h2rWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2rBias", this, ParameterType.BIAS),
                    new Parameter("i2zWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2zBias", this, ParameterType.BIAS),
                    new Parameter("h2zWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2zBias", this, ParameterType.BIAS),
                    new Parameter("i2nWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2nBias", this, ParameterType.BIAS),
                    new Parameter("h2nWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2nBias", this, ParameterType.BIAS));

    private Parameter state = new Parameter("state", this, ParameterType.OTHER);

    GRU(Builder builder) {
        super(builder);
        mode = "gru";
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
        List<Parameter> directParameters = new ArrayList<>(parameters);
        directParameters.add(state);
        return directParameters;
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(Shape[] inputShapes) {
        this.inputShapes = inputShapes;
        Shape inputShape = inputShapes[0];
        Block.validateLayout(EXPECTED_LAYOUT, inputShape.getLayout());
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        Shape shape = inputShapes[0];
        long channelSize = shape.get(2);
        long batchSize = shape.get(1);
        switch (name) {
            case "i2rWeight":
            case "i2zWeight":
            case "i2nWeight":
                return new Shape(stateSize, channelSize);
            case "h2rWeight":
            case "h2zWeight":
            case "h2nWeight":
                return new Shape(stateSize, stateSize);
            case "h2rBias":
            case "i2rBias":
            case "h2zBias":
            case "i2zBias":
            case "h2nBias":
            case "i2nBias":
                return new Shape(stateSize);
            case "state":
                return new Shape(numStackedLayers, batchSize, stateSize);
            default:
                throw new IllegalArgumentException("Invalid parameter name: " + name);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        for (Parameter parameter : parameters) {
            parameter.save(os);
        }
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
        for (Parameter parameter : parameters) {
            parameter.load(manager, is);
        }
        state.load(manager, is);
    }

    private NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        validateInputSize(inputs);
        NDArray head = inputs.head();
        Device device = head.getDevice();

        NDList result = new NDList(head);
        try (NDList parameterList = new NDList()) {
            for (Parameter parameter : parameters) {
                NDArray array = parameterStore.getValue(parameter, device);
                array = array.flatten();
                array.setName(parameter.getName());
                parameterList.add(array);
            }
            NDArray array = NDArrays.concat(parameterList);
            result.add(array);
            // array.waitToRead();
        }

        result.add(parameterStore.getValue(state, device));
        if (useSequenceLength) {
            result.add(inputs.get(1));
        }
        return result;
    }

    /** The Builder to construct a {@link GRU} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link GRU} block.
         *
         * @return the {@link GRU} block
         */
        public GRU build() {
            if (stateSize == -1 || numStackedLayers == -1) {
                throw new IllegalArgumentException("Must set stateSize and numStackedLayers");
            }
            return new GRU(this);
        }
    }
}
