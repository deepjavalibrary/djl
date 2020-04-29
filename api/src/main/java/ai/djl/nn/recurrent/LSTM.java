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
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * {@code LSTM} is an implementation of recurrent neural networks which applies Long Short-Term
 * Memory recurrent layer to input.
 *
 * <p>Reference paper - LONG SHORT-TERM MEMORY - Hochreiter, 1997.
 * http://www.bioinf.jku.at/publications/older/2604.pdf
 *
 * <p>The LSTM operator is formulated as below:
 *
 * <p>$$ \begin{split}\begin{array}{ll} i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi}
 * h_{(t-1)} + b_{hi}) \\ f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
 * g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\ o_t = \mathrm{sigmoid}(W_{io} x_t
 * + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\ c_t = f_t * c_{(t-1)} + i_t * g_t \\ h_t = o_t *
 * \tanh(c_t) \end{array}\end{split} $$
 */
public class LSTM extends RecurrentBlock {

    private static final byte VERSION = 2;

    private boolean clipLstmState;
    private double lstmStateClipMin;
    private double lstmStateClipMax;
    private NDArray beginStateCell;

    /**
     * Creates an LSTM block.
     *
     * @param builder the builder used to create the RNN block
     */
    LSTM(Builder builder) {
        super(builder);
        mode = "lstm";
        gates = 4;
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

        NDList output;
        if (clipLstmState) {
            output =
                    ex.lstm(
                            inputs,
                            stateSize,
                            dropRate,
                            numStackedLayers,
                            useSequenceLength,
                            isBidirectional(),
                            true,
                            lstmStateClipMin,
                            lstmStateClipMax,
                            params);
        } else {
            output =
                    ex.rnn(
                            inputs,
                            mode,
                            stateSize,
                            dropRate,
                            numStackedLayers,
                            useSequenceLength,
                            isBidirectional(),
                            true,
                            params);
        }

        NDList result = new NDList(output.head().transpose(1, 0, 2));
        if (stateOutputs) {
            result.add(output.get(1));
            result.add(output.get(2));
        }
        resetBeginStates();
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void setBeginStates(NDList beginStates) {
        this.beginState = beginStates.get(0);
        this.beginStateCell = beginStates.get(1);
    }

    @Override
    protected void resetBeginStates() {
        beginState = null;
        beginStateCell = null;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        validateInputSize(inputs);
        long batchSize = inputs.head().getShape().get(0);
        inputs = updateInputLayoutToTNC(inputs);
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
        // Adding state and stateCell
        Shape stateShape = new Shape(numStackedLayers * numDirections, batchSize, stateSize);
        if (beginState != null) {
            result.add(beginState);
            result.add(beginStateCell);
        } else {
            result.add(inputs.head().getManager().zeros(stateShape));
            result.add(inputs.head().getManager().zeros(stateShape));
        }
        if (useSequenceLength) {
            result.add(inputs.get(1));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        saveInputShapes(os);
        for (Parameter parameter : parameters) {
            parameter.save(os);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version == VERSION) {
            readInputShapes(is);
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        for (Parameter parameter : parameters) {
            parameter.load(manager, is);
        }
    }

    /**
     * Creates a builder to build a {@link LSTM}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link LSTM} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the minimum and maximum clip value of LSTM states.
         *
         * @param lstmStateClipMin the minimum clip value of LSTM states
         * @param lstmStateClipMax the maximum clip value of LSTM states
         * @return this Builder
         */
        public Builder optLstmStateClipMin(float lstmStateClipMin, float lstmStateClipMax) {
            this.lstmStateClipMin = lstmStateClipMin;
            this.lstmStateClipMax = lstmStateClipMax;
            this.clipLstmState = true;
            return self();
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
