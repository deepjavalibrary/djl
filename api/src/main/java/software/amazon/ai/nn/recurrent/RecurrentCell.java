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

import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.recurrent.RNN.Activation;

public interface RecurrentCell extends Block {

    abstract class Builder<R extends RecurrentCell> {
        float dropRate;
        long stateSize = -1;
        int numStackedLayers = -1;
        double lstmStateClipMin;
        double lstmStateClipMax;
        boolean clipLstmState;
        boolean useSequenceLength;
        boolean useBidirectional;
        boolean stateOutputs;
        RNN.Activation activation;

        public float getDropRate() {
            return dropRate;
        }

        public long getStateSize() {
            return stateSize;
        }

        public int getNumStackedLayers() {
            return numStackedLayers;
        }

        public double getLstmStateClipMin() {
            return lstmStateClipMin;
        }

        public double getLstmStateClipMax() {
            return lstmStateClipMax;
        }

        public boolean isClipLstmState() {
            return clipLstmState;
        }

        public boolean isUseSequenceLength() {
            return useSequenceLength;
        }

        public boolean isUseBidirectional() {
            return useBidirectional;
        }

        public boolean isStateOutputs() {
            return stateOutputs;
        }

        public Activation getActivation() {
            return activation;
        }

        /**
         * Sets the drop rate of the dropout on the outputs of each RNN layer, except the last
         * layer.
         *
         * @param dropRate drop rate of the dropout
         * @return Returns this Builder
         */
        public Builder<R> setDropRate(float dropRate) {
            this.dropRate = dropRate;
            return this;
        }

        /**
         * Sets the minimum and maximum clip value of LSTM states.
         *
         * @param lstmStateClipMin Minimum clip value of LSTM states
         * @param lstmStateClipMax Maximum clip value of LSTM states
         * @return Returns this Builder
         */
        public Builder<R> setLstmStateClipMin(float lstmStateClipMin, float lstmStateClipMax) {
            this.lstmStateClipMin = lstmStateClipMin;
            this.lstmStateClipMax = lstmStateClipMax;
            this.clipLstmState = true;
            return this;
        }

        /**
         * Sets the <b>Required</b> size of the state for each layer.
         *
         * @param stateSize Number of convolution filter(channel)
         * @return Returns this Builder
         */
        public Builder<R> setStateSize(int stateSize) {
            this.stateSize = stateSize;
            return this;
        }

        /**
         * Sets the <b>Required</b> number of stacked layers.
         *
         * @param numStackedLayers Number of convolution filter(channel)
         * @return Returns this Builder
         */
        public Builder<R> setNumStackedLayers(int numStackedLayers) {
            this.numStackedLayers = numStackedLayers;
            return this;
        }

        /**
         * Sets the activation for the RNN - ReLu or Tanh
         *
         * @param activation Projection size.
         * @return Returns this Builder
         */
        public Builder<R> setActivation(RNN.Activation activation) {
            this.activation = activation;
            return this;
        }

        /**
         * Sets the optional parameter of whether to include an extra input parameter
         * sequence_length to specify variable length sequence.
         *
         * @param useSequenceLength Whether to use sequence length
         * @return Returns this Builder
         */
        public Builder<R> setSequenceLength(boolean useSequenceLength) {
            this.useSequenceLength = useSequenceLength;
            return this;
        }

        /**
         * Sets the optional parameter of whether to use bidirectional recurrent layers.
         *
         * @param useBidirectional Whether to use bidirectional recurrent layers
         * @return Returns this Builder
         */
        public Builder<R> setBirectional(boolean useBidirectional) {
            this.useBidirectional = useBidirectional;
            return this;
        }

        /**
         * Sets the optional parameter of whether to have the states as symbol outputs.
         *
         * @param stateOutputs Whether to have the states as symbol output
         * @return Returns this Builder
         */
        public Builder<R> setStateOutput(boolean stateOutputs) {
            this.stateOutputs = stateOutputs;
            return this;
        }

        /**
         * Returns the constructed {@code RecurrentCell}.
         *
         * @return Returns the constructed {@code RecurrentCell}
         * @throws IllegalArgumentException Thrown if all required parameters (outChannels) have not
         *     been set
         */
        public abstract R build();
    }
}
