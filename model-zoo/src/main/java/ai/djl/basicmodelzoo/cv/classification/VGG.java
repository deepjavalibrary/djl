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
 * VGG model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
 * https://arxiv.org/abs/1409.1556 paper.
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-modern/vgg.html">The D2L chapter on
 *     VGG</a>
 */
public final class VGG {

    private VGG() {}

    /**
     * Creates a VGG block with the help of the VGG Builder.
     *
     * @param builder the {@link VGG.Builder} with the necessary arguments
     * @return a VGG block.
     */
    public static Block vgg(Builder builder) {

        SequentialBlock block = new SequentialBlock();
        VGG vgg = new VGG();
        // The convolutional layer part
        for (int[] arr : builder.convArch) {
            block.add(vgg.vggBlock(arr[0], arr[1]));
        }

        // The fully connected layer part
        block.add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(4096).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.5f).build())
                .add(Linear.builder().setUnits(4096).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.5f).build())
                .add(Linear.builder().setUnits(10).build());

        return block;
    }

    /**
     * Creates a constituent VGG block that becomes a part of the whole VGG model.
     *
     * @param numConvs Numbers of layers in each feature block.
     * @param numChannels Numbers of filters in each feature block. List length should match the
     *     layers.
     * @return a constituent vgg block.
     */
    public SequentialBlock vggBlock(int numConvs, int numChannels) {

        SequentialBlock tempBlock = new SequentialBlock();
        for (int i = 0; i < numConvs; i++) {
            tempBlock
                    .add(
                            Conv2d.builder()
                                    .setFilters(numChannels)
                                    .setKernelShape(new Shape(3, 3))
                                    .optPadding(new Shape(1, 1))
                                    .build())
                    .add(Activation::relu);
        }
        tempBlock.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        return tempBlock;
    }

    /**
     * Creates a builder to build a {@link VGG}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link VGG} object. */
    public static final class Builder {

        int numLayers = 11;
        int[][] convArch = {{1, 64}, {1, 128}, {2, 256}, {2, 512}, {2, 512}};

        /**
         * Sets the number of layers. It is equal to total sum of numConvs in convArch + 3.
         *
         * @param numLayers the number of layers in the network. default is 11.
         * @return this {@code Builder}
         */
        public VGG.Builder setNumLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }

        /**
         * Sets the number of blocks according to the user. It can be of multiple types, VGG-11,
         * VGG-13, VGG-16, VGG-19.
         *
         * @param convArch 2-D array consisting of number of convolutions and the number of
         *     channels.
         * @return this {@code Builder}
         */
        public VGG.Builder setConvArch(int[][] convArch) {
            int numConvs = 0;
            for (int[] layer : convArch) {
                numConvs += layer[0];
            }

            if (numConvs != (numLayers - 3)) {
                throw new IllegalArgumentException(
                        "total sum of channels in the array "
                                + "should be equal to the ( numLayers - 3 )");
            }

            this.convArch = convArch;
            return this;
        }

        /**
         * Builds a {@link VGG} block.
         *
         * @return the {@link VGG} block
         */
        public Block build() {
            return vgg(this);
        }
    }
}
