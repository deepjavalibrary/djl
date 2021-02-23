/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;

/**
 * The model was introduced (and named for) Yann Lecun, for the purpose of recognizing handwritten
 * digits in images [LeNet5](http://yann.lecun.com/exdb/lenet/).
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-neural-networks/lenet.html">The D2L
 *     chapter on LeNet</a>
 */
public final class LeNet {

    private LeNet() {}

    /**
     * Creates a LeNet network block with the help of the LeNet Builder.
     *
     * @param builder the {@link LeNet.Builder} with the necessary arguments.
     * @return a LeNet block.
     */
    public static Block leNet(Builder builder) {

        return new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .optPadding(new Shape(2, 2))
                                .optBias(false)
                                .setFilters(builder.numChannels[0])
                                .build())
                .add(Activation::sigmoid)
                .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .setFilters(builder.numChannels[1])
                                .build())
                .add(Activation::sigmoid)
                .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                // Blocks.batchFlattenBlock() will transform the input of the shape (batch
                // size, channel,
                // height, width) into the input of the shape (batch size,
                // channel * height * width)
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(builder.numChannels[2]).build())
                .add(Activation::sigmoid)
                .add(Linear.builder().setUnits(builder.numChannels[3]).build())
                .add(Activation::sigmoid)
                .add(Linear.builder().setUnits(10).build());
    }

    /**
     * Creates a builder to build a {@link LeNet}.
     *
     * @return a new builder
     */
    public static LeNet.Builder builder() {
        return new LeNet.Builder();
    }

    /** The Builder to construct a {@link LeNet} object. */
    public static final class Builder {

        int numLayers = 4;
        int[] numChannels = {6, 16, 120, 84};

        Builder() {}

        /**
         * Sets the number of channels for the LeNet blocks.
         *
         * @param numChannels the number of channels for every LeNet block.
         * @return this {@code Builder}
         */
        public LeNet.Builder setNumChannels(int[] numChannels) {

            if (numChannels.length != numLayers) {
                throw new IllegalArgumentException(
                        "number of channels can be equal to " + numLayers);
            }

            this.numChannels = numChannels;
            return this;
        }

        /**
         * Builds a {@link LeNet} block.
         *
         * @return the {@link LeNet} block
         */
        public Block build() {
            return leNet(this);
        }
    }
}
