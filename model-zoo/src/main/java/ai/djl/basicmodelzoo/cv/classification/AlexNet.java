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
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

/**
 * {@code AlexNet} contains a generic implementation of AlexNet adapted from [torchvision
 * implmentation](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)
 *
 * <p>AlexNet model from the "One weird trick..." https://arxiv.org/abs/1404.5997 paper.
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-modern/alexnet.html">The D2L chapter on
 *     AlexNet</a>
 */
public final class AlexNet {

    private AlexNet() {}

    /**
     * Creates a AlexNet network block with the help of the AlexNet Builder.
     *
     * @param builder the {@link AlexNet.Builder} with the necessary arguments.
     * @return a AlexNet block.
     */
    public static Block alexNet(Builder builder) {
        return new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(11, 11))
                                .optStride(new Shape(4, 4))
                                .setFilters(builder.numChannels[0])
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Make the convolution window smaller, set padding to 2 for consistent
                // height and width across the input and output, and increase the
                // number of output channels
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .optPadding(new Shape(2, 2))
                                .setFilters(builder.numChannels[1])
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Use three successive convolutional layers and a smaller convolution
                // window. Except for the final convolutional layer, the number of
                // output channels is further increased. Pooling layers are not used to
                // reduce the height and width of input after the first two
                // convolutional layers
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .setFilters(builder.numChannels[2])
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .setFilters(builder.numChannels[3])
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .setFilters(builder.numChannels[4])
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Here, the number of outputs of the fully connected layer is several
                // times larger than that in LeNet. Use the dropout layer to mitigate
                // over fitting
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(builder.numChannels[5]).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(builder.dropOutRate).build())
                .add(Linear.builder().setUnits(builder.numChannels[6]).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(builder.dropOutRate).build())
                // Output layer. The number of
                // classes is 10, instead of 1000 as in the paper
                .add(Linear.builder().setUnits(10).build());
    }

    /**
     * Creates a builder to build a {@link AlexNet}.
     *
     * @return a new builder
     */
    public static AlexNet.Builder builder() {
        return new AlexNet.Builder();
    }

    /** The Builder to construct a {@link AlexNet} object. */
    public static final class Builder {

        float dropOutRate = 0.5f;
        int numLayers = 7;
        int[] numChannels = {96, 256, 384, 384, 256, 4096, 4096};

        Builder() {}

        /**
         * Sets the dropout rate in the network.
         *
         * @param dropOutRate the dropout rate
         * @return this {@code Builder}
         */
        public AlexNet.Builder setDropOutRate(float dropOutRate) {
            this.dropOutRate = dropOutRate;
            return this;
        }

        /**
         * Sets the number of channels for the AlexNet blocks.
         *
         * @param numChannels the number of channels for every AlexNet block.
         * @return this {@code Builder}
         */
        public AlexNet.Builder setNumChannels(int[] numChannels) {

            if (numChannels.length != numLayers) {
                throw new IllegalArgumentException(
                        "number of channels should be equal to " + numLayers);
            }

            this.numChannels = numChannels;
            return this;
        }

        /**
         * Builds a {@link AlexNet} block.
         *
         * @return the {@link AlexNet} block
         */
        public Block build() {
            return alexNet(this);
        }
    }
}
