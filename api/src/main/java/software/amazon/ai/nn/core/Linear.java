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
package software.amazon.ai.nn.core;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.BlockFactory;

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
public interface Linear extends Block {

    NDArray forward(NDArray data);

    /** The Builder to construct a {@link Linear} type of {@link Block}. */
    class Builder {

        private BlockFactory factory;
        private long outChannels;
        private boolean bias = true;

        public long getOutChannels() {
            return outChannels;
        }

        public Builder setFactory(BlockFactory factory) {
            this.factory = factory;
            return this;
        }

        /**
         * Sets the <b>Required</b> number of output channels.
         *
         * @param outChannels Number of desired output channels
         * @return Returns this Builder
         */
        public Builder setOutChannels(long outChannels) {
            this.outChannels = outChannels;
            return this;
        }

        public boolean isBias() {
            return bias;
        }

        /**
         * Sets the optional parameter of whether to include a bias vector with default of true.
         *
         * @param bias Whether to use a bias vector parameter
         * @return Returns this Builder
         */
        public Builder setBias(boolean bias) {
            this.bias = bias;
            return this;
        }

        /**
         * Returns the constructed {@code Linear}.
         *
         * @return Returns the constructed {@code Linear}
         * @throws IllegalArgumentException Thrown if all required parameters (outChannels) have not
         *     been set
         */
        public Linear build() {
            if (outChannels == 0) {
                throw new IllegalArgumentException("You must specify outChannels");
            }
            return factory.createLinear(this);
        }
    }
}
