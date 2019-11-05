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
 * Applies Long Short-Term Memory recurrent layer to input.
 *
 * <p>Reference paper - LONG SHORT-TERM MEMORY - Hochreiter, 1997.
 * http://www.bioinf.jku.at/publications/older/2604.pdf
 *
 * <p>$$ \begin{split}\begin{array}{ll} i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi}
 * h_{(t-1)} + b_{hi}) \\ f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
 * g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\ o_t = \mathrm{sigmoid}(W_{io} x_t
 * + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\ c_t = f_t * c_{(t-1)} + i_t * g_t \\ h_t = o_t *
 * \tanh(c_t) \end{array}\end{split} $$
 */
public class LSTM extends RecurrentCell {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.TIME, LayoutType.BATCH, LayoutType.CHANNEL
    };

    private static final byte VERSION = 1;

    private boolean clipLstmState;
    private double lstmStateClipMin;
    private double lstmStateClipMax;

    private List<Parameter> parameters =
            Arrays.asList(
                    new Parameter("i2iWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2iBias", this, ParameterType.BIAS),
                    new Parameter("h2iWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2iBias", this, ParameterType.BIAS),
                    new Parameter("i2fWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2fBias", this, ParameterType.BIAS),
                    new Parameter("h2fWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2fBias", this, ParameterType.BIAS),
                    new Parameter("i2gWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2gBias", this, ParameterType.BIAS),
                    new Parameter("h2gWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2gBias", this, ParameterType.BIAS),
                    new Parameter("i2oWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2oBias", this, ParameterType.BIAS),
                    new Parameter("h2oWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2oBias", this, ParameterType.BIAS));

    private Parameter state = new Parameter("state", this, ParameterType.OTHER);
    private Parameter stateCell = new Parameter("state_cell", this, ParameterType.OTHER);

    /**
     * Creates an LSTM block.
     *
     * @param builder the builder used to create the RNN block
     */
    LSTM(Builder builder) {
        super(builder);
        mode = "lstm";
        clipLstmState = builder.clipLstmState;
        lstmStateClipMin = builder.lstmStateClipMin;
        lstmStateClipMax = builder.lstmStateClipMax;
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(parameterStore, inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();

        if (clipLstmState) {
            return ex.lstm(
                    inputs,
                    stateSize,
                    dropRate,
                    numStackedLayers,
                    useSequenceLength,
                    useBidirectional,
                    stateOutputs,
                    lstmStateClipMin,
                    lstmStateClipMax,
                    params);
        }

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
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        return new Shape[] {new Shape(inputShape.get(0), inputShape.get(1), stateSize)};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        List<Parameter> directParameters = new ArrayList<>(this.parameters);
        directParameters.add(state);
        directParameters.add(stateCell);
        return directParameters;
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
            case "i2iWeight":
            case "i2fWeight":
            case "i2gWeight":
            case "i2oWeight":
                return new Shape(stateSize, channelSize);
            case "h2iWeight":
            case "h2fWeight":
            case "h2gWeight":
            case "h2oWeight":
                return new Shape(stateSize, stateSize);
            case "h2iBias":
            case "i2iBias":
            case "h2fBias":
            case "i2fBias":
            case "h2gBias":
            case "i2gBias":
            case "h2oBias":
            case "i2oBias":
                return new Shape(stateSize);
            case "state":
            case "state_cell":
                return new Shape(numStackedLayers, batchSize, stateSize);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    private NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        validateInputSize(inputs);
        NDArray head = inputs.head();
        Device device = head.getDevice();

        NDList result = new NDList(head);
        try (NDList parameterList = new NDList()) {
            for (Parameter parameter : parameters) {
                NDArray array = parameterStore.getValue(parameter, device);
                parameterList.add(array.flatten());
            }
            NDArray array = NDArrays.concat(parameterList);
            result.add(array);
        }
        result.add(parameterStore.getValue(state, device));
        result.add(parameterStore.getValue(stateCell, device));
        if (useSequenceLength) {
            result.add(inputs.get(1));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        for (Parameter parameter : parameters) {
            parameter.save(os);
        }
        state.save(os);
        stateCell.save(os);
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
        stateCell.load(manager, is);
    }

    /** The Builder to construct a {@link LSTM} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link LSTM} block.
         *
         * @return the {@link LSTM} block
         */
        public LSTM build() {
            if (stateSize == -1 || numStackedLayers == -1) {
                throw new IllegalArgumentException("Must set stateSize and numStackedLayers");
            }
            return new LSTM(this);
        }
    }
}
