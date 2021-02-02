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

    /**
     * Creates an LSTM block.
     *
     * @param builder the builder used to create the RNN block
     */
    LSTM(Builder builder) {
        super(builder);
        gates = 4;
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
            Shape stateShape =
                    new Shape(
                            (long) numLayers * getNumDirections(),
                            input.size(batchIndex),
                            stateSize);
            // hidden state
            inputs.add(input.getManager().zeros(stateShape));
            // cell
            inputs.add(input.getManager().zeros(stateShape));
        }
        NDList outputs =
                ex.lstm(
                        input,
                        new NDList(inputs.get(1), inputs.get(2)),
                        rnnParams,
                        hasBiases,
                        numLayers,
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
         * Builds a {@link LSTM} block.
         *
         * @return the {@link LSTM} block
         */
        public LSTM build() {
            Preconditions.checkArgument(
                    stateSize > 0 && numLayers > 0, "Must set stateSize and numStackedLayers");
            return new LSTM(this);
        }
    }
}
