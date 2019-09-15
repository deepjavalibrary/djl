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

import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.AbstractBlock;
import software.amazon.ai.nn.BlockFactory;
import software.amazon.ai.nn.recurrent.RNN.Activation;

public abstract class RecurrentCell extends AbstractBlock {

    protected long stateSize;
    protected float dropRate;
    protected int numStackedLayers;
    protected String mode;
    protected boolean useSequenceLength;
    protected boolean useBidirectional;
    protected boolean stateOutputs;

    public RecurrentCell(NDManager manager) {
        super(manager);
    }

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected BlockFactory factory;
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

        public T setFactory(BlockFactory factory) {
            this.factory = factory;
            return self();
        }

        /**
         * Sets the drop rate of the dropout on the outputs of each RNN layer, except the last
         * layer.
         *
         * @param dropRate drop rate of the dropout
         * @return Returns this Builder
         */
        public T setDropRate(float dropRate) {
            this.dropRate = dropRate;
            return self();
        }

        /**
         * Sets the minimum and maximum clip value of LSTM states.
         *
         * @param lstmStateClipMin Minimum clip value of LSTM states
         * @param lstmStateClipMax Maximum clip value of LSTM states
         * @return Returns this Builder
         */
        public T setLstmStateClipMin(float lstmStateClipMin, float lstmStateClipMax) {
            this.lstmStateClipMin = lstmStateClipMin;
            this.lstmStateClipMax = lstmStateClipMax;
            this.clipLstmState = true;
            return self();
        }

        /**
         * Sets the <b>Required</b> size of the state for each layer.
         *
         * @param stateSize Number of convolution filter(channel)
         * @return Returns this Builder
         */
        public T setStateSize(int stateSize) {
            this.stateSize = stateSize;
            return self();
        }

        /**
         * Sets the <b>Required</b> number of stacked layers.
         *
         * @param numStackedLayers Number of convolution filter(channel)
         * @return Returns this Builder
         */
        public T setNumStackedLayers(int numStackedLayers) {
            this.numStackedLayers = numStackedLayers;
            return self();
        }

        /**
         * Sets the activation for the RNN - ReLu or Tanh.
         *
         * @param activation Projection size.
         * @return Returns this Builder
         */
        public T setActivation(RNN.Activation activation) {
            this.activation = activation;
            return self();
        }

        /**
         * Sets the optional parameter of whether to include an extra input parameter
         * sequence_length to specify variable length sequence.
         *
         * @param useSequenceLength Whether to use sequence length
         * @return Returns this Builder
         */
        public T setSequenceLength(boolean useSequenceLength) {
            this.useSequenceLength = useSequenceLength;
            return self();
        }

        /**
         * Sets the optional parameter of whether to use bidirectional recurrent layers.
         *
         * @param useBidirectional Whether to use bidirectional recurrent layers
         * @return Returns this Builder
         */
        public T setBirectional(boolean useBidirectional) {
            this.useBidirectional = useBidirectional;
            return self();
        }

        /**
         * Sets the optional parameter of whether to have the states as symbol outputs.
         *
         * @param stateOutputs Whether to have the states as symbol output
         * @return Returns this Builder
         */
        public T setStateOutput(boolean stateOutputs) {
            this.stateOutputs = stateOutputs;
            return self();
        }

        protected abstract T self();
    }
}
