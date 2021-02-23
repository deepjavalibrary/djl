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
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

/**
 * NiN uses convolutional layers with window shapes of 11×11 , 5×5 , and 3×3 , and the corresponding
 * numbers of output channels are the same as in AlexNet. Each NiN block is followed by a maximum
 * pooling layer with a stride of 2 and a window shape of 3×3 .
 *
 * <p>The conventional convolutional layer uses linear filters followed by a nonlinear activation
 * function to scan the input.
 *
 * <p>NiN model from the "Network In Network" http://arxiv.org/abs/1312.4400 paper.
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-modern/nin.html">The D2L chapter on
 *     NiN</a>
 */
public final class NiN {

    private NiN() {}

    /**
     * The NiN block consists of one convolutional layer followed by two 1×1 convolutional layers
     * that act as per-pixel fully-connected layers with ReLU activations. The convolution width of
     * the first layer is typically set by the user. The subsequent widths are fixed to 1×1.
     *
     * @param builder the {@link NiN.Builder} with the necessary arguments.
     * @return a NiN block.
     */
    public static Block niN(Builder builder) {

        NiN nin = new NiN();
        return new SequentialBlock()
                .add(
                        nin.niNBlock(
                                builder.numChannels[0],
                                new Shape(11, 11),
                                new Shape(4, 4),
                                new Shape(0, 0)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(
                        nin.niNBlock(
                                builder.numChannels[1],
                                new Shape(5, 5),
                                new Shape(1, 1),
                                new Shape(2, 2)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(
                        nin.niNBlock(
                                builder.numChannels[2],
                                new Shape(3, 3),
                                new Shape(1, 1),
                                new Shape(1, 1)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(Dropout.builder().optRate(builder.dropOutRate).build())
                .add(
                        nin.niNBlock(
                                builder.numChannels[3],
                                new Shape(3, 3),
                                new Shape(1, 1),
                                new Shape(1, 1)))
                // The global average pooling layer automatically sets the window shape
                // to the height and width of the input
                .add(Pool.globalAvgPool2dBlock())
                // Transform the four-dimensional output into two-dimensional output
                // with a shape of (batch size, 10)
                .add(Blocks.batchFlattenBlock());
    }

    /**
     * Creates a builder to build a {@link NiN}.
     *
     * @return a new builder
     */
    public static NiN.Builder builder() {
        return new Builder();
    }

    /**
     * Creates a constituent NiN block that becomes a part of the whole NiN model.
     *
     * @param numChannels the number of channels in a NiN block.
     * @param kernelShape kernel Shape in the 1st convolutional layer of a NiN block.
     * @param strideShape stride Shape in a NiN block.
     * @param paddingShape padding Shape in a NiN block.
     * @return a constituent niN block.
     */
    public SequentialBlock niNBlock(
            int numChannels, Shape kernelShape, Shape strideShape, Shape paddingShape) {

        return new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setKernelShape(kernelShape)
                                .optStride(strideShape)
                                .optPadding(paddingShape)
                                .setFilters(numChannels)
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(1, 1))
                                .setFilters(numChannels)
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(1, 1))
                                .setFilters(numChannels)
                                .build())
                .add(Activation::relu);
    }

    /** The Builder to construct a {@link NiN} object. */
    public static final class Builder {

        int numLayers = 4;
        int[] numChannels = {96, 256, 384, 10};
        float dropOutRate = 0.5f;

        Builder() {}

        /**
         * Sets the dropout rate in the network.
         *
         * @param dropOutRate the dropout rate
         * @return this {@code Builder}
         */
        public NiN.Builder setDropOutRate(float dropOutRate) {
            this.dropOutRate = dropOutRate;
            return this;
        }

        /**
         * Sets the number of channels for the niN blocks.
         *
         * @param numChannels the number of channels for every niN block.
         * @return this {@code Builder}
         */
        public NiN.Builder setNumChannels(int[] numChannels) {

            if (numChannels.length != numLayers) {
                throw new IllegalArgumentException(
                        "number of channels can be equal to " + numLayers);
            }

            this.numChannels = numChannels;
            return this;
        }

        /**
         * Builds a {@link NiN} block.
         *
         * @return the {@link NiN} block
         */
        public Block build() {
            return niN(this);
        }
    }
}
