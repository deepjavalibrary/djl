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
package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import java.util.Arrays;

/**
 * {@code ResNetV1} contains a generic implementation of ResNet adapted from
 * https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py (Original author Wei Wu) by
 * Antti-Pekka Hynninen.
 *
 * <p>Implementing the original resnet ILSVRC 2015 winning network from Kaiming He, Xiangyu Zhang,
 * Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-modern/resnet.html">The D2L chapter on
 *     ResNet</a>
 */
public final class ResNetV1 {

    private ResNetV1() {}

    /**
     * Builds a {@link Block} that represents a residual unit used in the implementation of the
     * Resnet model.
     *
     * @param numFilters the number of output channels
     * @param stride the stride of the convolution in each dimension
     * @param dimMatch whether the number of channels between input and output has to remain the
     *     same
     * @param bottleneck whether to use bottleneck architecture
     * @param batchNormMomentum the momentum to be used for {@link BatchNorm}
     * @return a {@link Block} that represents a residual unit
     */
    public static Block residualUnit(
            int numFilters,
            final Shape stride,
            final boolean dimMatch,
            boolean bottleneck,
            float batchNormMomentum) {
        SequentialBlock resUnit = new SequentialBlock();
        if (bottleneck) {
            resUnit.add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(1, 1))
                                    .setFilters(numFilters / 4)
                                    .optStride(stride)
                                    .optPadding(new Shape(0, 0))
                                    .optBias(true)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(1e-5f)
                                    .optMomentum(batchNormMomentum)
                                    .build())
                    .add(Activation::relu)
                    .add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(3, 3))
                                    .setFilters(numFilters / 4)
                                    .optStride(new Shape(1, 1))
                                    .optPadding(new Shape(1, 1))
                                    .optBias(false)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(2E-5f)
                                    .optMomentum(batchNormMomentum)
                                    .build())
                    .add(Activation::relu)
                    .add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(1, 1))
                                    .setFilters(numFilters)
                                    .optStride(new Shape(1, 1))
                                    .optPadding(new Shape(0, 0))
                                    .optBias(true)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(1E-5f)
                                    .optMomentum(batchNormMomentum)
                                    .build());

        } else {
            resUnit.add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(3, 3))
                                    .setFilters(numFilters)
                                    .optStride(stride)
                                    .optPadding(new Shape(1, 1))
                                    .optBias(false)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(1E-5f)
                                    .optMomentum(batchNormMomentum)
                                    .build())
                    .add(Activation::relu)
                    .add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(3, 3))
                                    .setFilters(numFilters)
                                    .optStride(new Shape(1, 1))
                                    .optPadding(new Shape(1, 1))
                                    .optBias(false)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(1E-5f)
                                    .optMomentum(batchNormMomentum)
                                    .build());
        }
        SequentialBlock shortcut = new SequentialBlock();
        if (dimMatch) {
            shortcut.add(Blocks.identityBlock());
        } else {
            shortcut.add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(1, 1))
                                    .setFilters(numFilters)
                                    .optStride(stride)
                                    .optPadding(new Shape(0, 0))
                                    .optBias(false)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(1E-5f)
                                    .optMomentum(batchNormMomentum)
                                    .build());
        }
        return new ParallelBlock(
                list -> {
                    NDList unit = list.get(0);
                    NDList parallel = list.get(1);
                    return new NDList(
                            unit.singletonOrThrow()
                                    .add(parallel.singletonOrThrow())
                                    .getNDArrayInternal()
                                    .relu());
                },
                Arrays.asList(resUnit, shortcut));
    }

    /**
     * Creates a new {@link Block} of {@code ResNetV1} with the arguments from the given {@link
     * Builder}.
     *
     * @param builder the {@link Builder} with the necessary arguments
     * @return a {@link Block} that represents the required ResNet model
     */
    public static Block resnet(Builder builder) {
        int numStages = builder.units.length;
        long height = builder.imageShape.get(1);
        SequentialBlock resNet = new SequentialBlock();
        if (height <= 32) {
            resNet.add(
                    Conv2d.builder()
                            .setKernelShape(new Shape(3, 3))
                            .setFilters(builder.filters[0])
                            .optStride(new Shape(1, 1))
                            .optPadding(new Shape(1, 1))
                            .optBias(false)
                            .build());
        } else {
            resNet.add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(7, 7))
                                    .setFilters(builder.filters[0])
                                    .optStride(new Shape(2, 2))
                                    .optPadding(new Shape(3, 3))
                                    .optBias(false)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(2E-5f)
                                    .optMomentum(builder.batchNormMomentum)
                                    .build())
                    .add(Activation.reluBlock())
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));
        }
        Shape resStride = new Shape(1, 1);
        for (int i = 0; i < numStages; i++) {
            resNet.add(
                    residualUnit(
                            builder.filters[i + 1],
                            resStride,
                            false,
                            builder.bottleneck,
                            builder.batchNormMomentum));
            for (int j = 0; j < builder.units[i] - 1; j++) {
                resNet.add(
                        residualUnit(
                                builder.filters[i + 1],
                                new Shape(1, 1),
                                true,
                                builder.bottleneck,
                                builder.batchNormMomentum));
            }
            if (i == 0) {
                resStride = new Shape(2, 2);
            }
        }
        return resNet.add(Pool.globalAvgPool2dBlock())
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(builder.outSize).build())
                .add(Blocks.batchFlattenBlock());
    }

    /**
     * Creates a builder to build a {@link ResNetV1}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link ResNetV1} object. */
    public static final class Builder {

        int numLayers;
        int numStages;
        long outSize;
        float batchNormMomentum = 0.9f;
        Shape imageShape;
        boolean bottleneck;
        int[] units;
        int[] filters;

        Builder() {}

        /**
         * Sets the number of layers in the network.
         *
         * @param numLayers the number of layers
         * @return this {@code Builder}
         */
        public Builder setNumLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }

        /**
         * Sets the size of the output.
         *
         * @param outSize the output size
         * @return this {@code Builder}
         */
        public Builder setOutSize(long outSize) {
            this.outSize = outSize;
            return this;
        }

        /**
         * Sets the momentum of batchNorm layer.
         *
         * @param batchNormMomentum the momentum
         * @return this {@code Builder}
         */
        public Builder optBatchNormMomentum(float batchNormMomentum) {
            this.batchNormMomentum = batchNormMomentum;
            return this;
        }

        /**
         * Sets the shape of the image.
         *
         * @param imageShape the shape of the image
         * @return this {@code Builder}
         */
        public Builder setImageShape(Shape imageShape) {
            this.imageShape = imageShape;
            return this;
        }

        /**
         * Builds a {@link ResNetV1} block.
         *
         * @return the {@link ResNetV1} block
         */
        public Block build() {
            if (imageShape == null) {
                throw new IllegalArgumentException("Must set imageShape");
            }
            long height = imageShape.get(1);
            if (height <= 28) {
                numStages = 3;
                int perUnit;
                if ((numLayers - 2) % 9 == 0 && numLayers >= 164) {
                    perUnit = (numLayers - 2) / 9;
                    filters = new int[] {16, 64, 128, 256};
                    bottleneck = true;
                } else if ((numLayers - 2) % 6 == 0 && numLayers < 164) {
                    perUnit = (numLayers - 2) / 6;
                    filters = new int[] {16, 16, 32, 64};
                    bottleneck = false;
                } else {
                    throw new IllegalArgumentException(
                            "no experiments done on num_layers "
                                    + numLayers
                                    + ", you can do it yourself");
                }
                units = new int[numStages];
                for (int i = 0; i < numStages; i++) {
                    units[i] = perUnit;
                }
            } else {
                numStages = 4;
                if (numLayers >= 50) {
                    filters = new int[] {64, 256, 512, 1024, 2048};
                    bottleneck = true;
                } else {
                    filters = new int[] {64, 64, 128, 256, 512};
                    bottleneck = true;
                }
                if (numLayers == 18) {
                    units = new int[] {2, 2, 2, 2};
                } else if (numLayers == 34) {
                    units = new int[] {3, 4, 6, 3};
                } else if (numLayers == 50) {
                    units = new int[] {3, 4, 6, 3};
                } else if (numLayers == 101) {
                    units = new int[] {3, 4, 23, 3};
                } else if (numLayers == 152) {
                    units = new int[] {3, 8, 36, 3};
                } else if (numLayers == 200) {
                    units = new int[] {3, 24, 36, 3};
                } else if (numLayers == 269) {
                    units = new int[] {3, 30, 48, 8};
                } else {
                    throw new IllegalArgumentException(
                            "no experiments done on num_layers "
                                    + numLayers
                                    + ", you can do it yourself");
                }
            }
            return resnet(this);
        }
    }
}
