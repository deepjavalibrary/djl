/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.nn.convolutional;

import software.amazon.ai.Block;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.types.Shape;

public interface Convolution extends Block {
    NDArray forward(NDArray data);

    abstract class Builder {

        Shape kernel;
        Shape stride;
        Shape pad;
        Shape dilate;
        int numFilters;
        int numGroups = 1;
        boolean noBias;

        /**
         * Sets the shape of the kernel.
         *
         * @param kernel Shape of the kernel
         * @return Returns this Builder
         */
        public Builder setKernel(Shape kernel) {
            this.kernel = kernel;
            return this;
        }

        /**
         * Sets the stride of the convolution. Defaults to 1 in each dimension.
         *
         * @param stride Shape of the stride
         * @return Returns this Builder
         */
        public Builder setStride(Shape stride) {
            this.stride = stride;
            return this;
        }

        /**
         * Sets the padding along each dimension. Defaults to zero along each dimension
         *
         * @param pad Padding along each dimension
         * @return Returns this Builder
         */
        public Builder setPad(Shape pad) {
            this.pad = pad;
            return this;
        }

        /**
         * Sets the padding along each dimension. Defaults to zero along each dimension
         *
         * @param dilate Padding along each dimension
         * @return Returns this Builder
         */
        public Builder setDilate(Shape dilate) {
            this.dilate = dilate;
            return this;
        }

        /**
         * Sets the <b>Required</b> number of filters.
         *
         * @param numFilters Number of convolution filter(channel)
         * @return Returns this Builder
         */
        public Builder setNumFilters(int numFilters) {
            this.numFilters = numFilters;
            return this;
        }

        /**
         * Sets the number of group partitions.
         *
         * @param numGroups Number of group partitions
         * @return Returns this Builder
         */
        public Builder setNumGroups(int numGroups) {
            this.numGroups = numGroups;
            return this;
        }

        /**
         * Sets the optional parameter of whether to include a bias vector with default of true.
         *
         * @param noBias Whether to use a bias vector parameter
         * @return Returns this Builder
         */
        public Builder setBias(boolean noBias) {
            this.noBias = noBias;
            return this;
        }

        /**
         * Returns the constructed {@code Convolution}.
         *
         * @return Returns the constructed {@code Convolution}
         * @throws IllegalArgumentException Thrown if all required parameters (outChannels) have not
         *     been set
         */
        public abstract Convolution build();
    }
}
