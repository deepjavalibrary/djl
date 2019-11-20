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

import ai.djl.ndarray.NDList;
import ai.djl.nn.ParameterBlock;

/**
 * Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are implemented,
 * with both multi-layer and bidirectional support.
 */
public abstract class RecurrentCell extends ParameterBlock {

    protected long stateSize;
    protected float dropRate;
    protected int numStackedLayers;
    protected String mode;
    protected boolean useSequenceLength;
    protected boolean useBidirectional;
    protected boolean stateOutputs;

    /**
     * Creates a {@code RecurrentCell} object.
     *
     * @param builder the {@code Builder} that has the necessary configurations
     */
    public RecurrentCell(BaseBuilder<?> builder) {
        stateSize = builder.stateSize;
        dropRate = builder.dropRate;
        numStackedLayers = builder.numStackedLayers;
        useSequenceLength = builder.useSequenceLength;
        useBidirectional = builder.useBidirectional;
        stateOutputs = builder.stateOutputs;
    }

    protected void validateInputSize(NDList inputs) {
        int numberofInputsRequired = 1;
        if (useSequenceLength) {
            numberofInputsRequired = 2;
        }
        if (inputs.size() != numberofInputsRequired) {
            throw new IllegalArgumentException(
                    "Invalid number of inputs for RNN. Size of input NDList must be "
                            + numberofInputsRequired
                            + " when useSequenceLength is "
                            + useSequenceLength);
        }
    }

    /** The Builder to construct a {@link RecurrentCell} type of {@link ai.djl.nn.Block}. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected float dropRate;
        protected long stateSize = -1;
        protected int numStackedLayers = -1;
        protected double lstmStateClipMin;
        protected double lstmStateClipMax;
        protected boolean clipLstmState;
        protected boolean useSequenceLength;
        protected boolean useBidirectional;
        protected boolean stateOutputs;
        protected RNN.Activation activation;

        /**
         * Sets the drop rate of the dropout on the outputs of each RNN layer, except the last
         * layer.
         *
         * @param dropRate the drop rate of the dropout
         * @return this Builder
         */
        public T optDropRate(float dropRate) {
            this.dropRate = dropRate;
            return self();
        }

        /**
         * Sets the minimum and maximum clip value of LSTM states.
         *
         * @param lstmStateClipMin the minimum clip value of LSTM states
         * @param lstmStateClipMax the maximum clip value of LSTM states
         * @return this Builder
         */
        public T optLstmStateClipMin(float lstmStateClipMin, float lstmStateClipMax) {
            this.lstmStateClipMin = lstmStateClipMin;
            this.lstmStateClipMax = lstmStateClipMax;
            this.clipLstmState = true;
            return self();
        }

        /**
         * Sets the <b>Required</b> size of the state for each layer.
         *
         * @param stateSize the size of the state for each layer
         * @return this Builder
         */
        public T setStateSize(int stateSize) {
            this.stateSize = stateSize;
            return self();
        }

        /**
         * Sets the <b>Required</b> number of stacked layers.
         *
         * @param numStackedLayers the number of stacked layers
         * @return this Builder
         */
        public T setNumStackedLayers(int numStackedLayers) {
            this.numStackedLayers = numStackedLayers;
            return self();
        }

        /**
         * Sets the activation for the RNN - ReLu or Tanh.
         *
         * @param activation the activation
         * @return this Builder
         */
        public T setActivation(RNN.Activation activation) {
            this.activation = activation;
            return self();
        }

        /**
         * Sets the optional parameter that indicates whether to include an extra input parameter
         * sequence_length to specify variable length sequence.
         *
         * @param useSequenceLength whether to use sequence length
         * @return this Builder
         */
        public T setSequenceLength(boolean useSequenceLength) {
            this.useSequenceLength = useSequenceLength;
            return self();
        }

        /**
         * Sets the optional parameter that indicates whether to use bidirectional recurrent layers.
         *
         * @param useBidirectional whether to use bidirectional recurrent layers
         * @return this Builder
         */
        public T optBidrectional(boolean useBidirectional) {
            this.useBidirectional = useBidirectional;
            return self();
        }

        /**
         * Sets the optional parameter that indicates whether to have the states as symbol outputs.
         *
         * @param stateOutputs whether to have the states as symbol output
         * @return this Builder
         */
        public T optStateOutput(boolean stateOutputs) {
            this.stateOutputs = stateOutputs;
            return self();
        }

        protected abstract T self();
    }
}
