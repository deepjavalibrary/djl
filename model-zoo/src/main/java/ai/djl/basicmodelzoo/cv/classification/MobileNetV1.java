/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;

/**
 * {@code MobileNetV1} contains a generic implementation of Mobilenet adapted from
 * https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenet.py (Original author
 * weiaicunzai).
 *
 * <p>see https://arxiv.org/pdf/1704.04861.pdf for more information about MobileNet
 */
public final class MobileNetV1 {
    static final int[] FILTERS = {32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024};

    private MobileNetV1() {}

    /**
     * Builds a {@link Block} that represent a depthWise-pointWise Unit used in the implementation
     * of the MobileNet Model.
     *
     * @param inputChannels number of inputChannels, used for depthWise Kernel
     * @param outputChannels number of outputChannels, used for pointWise kernel
     * @param stride control the stride of depthWise Kernel
     * @param builder add the builder to obtain batchNormMomentum
     * @return a {@link Block} that represent a depthWise-pointWise Unit
     */
    public static Block depthSeparableConv2d(
            int inputChannels, int outputChannels, int stride, Builder builder) {
        // depthWise does not include bias
        SequentialBlock depthWise = new SequentialBlock();
        depthWise
                .add(
                        Conv2d.builder()
                                .setKernelShape(
                                        new Shape(3, 3)) // the kernel size of depthWise is always 3
                                .optBias(false)
                                .optPadding(new Shape(1, 1)) // padding = same
                                .optStride(new Shape(stride, stride)) // stride is either 2 or 1
                                .optGroups(
                                        inputChannels) // depthWise with 1 filter per input channel
                                .setFilters(inputChannels)
                                .build())
                .add( // add a batchNorm
                        BatchNorm.builder()
                                .optEpsilon(2E-5f)
                                .optMomentum(builder.batchNormMomentum)
                                .build())
                .add(Activation.reluBlock());

        SequentialBlock pointWise = new SequentialBlock();
        pointWise
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(1, 1)) // no padding or stride
                                .setFilters(outputChannels)
                                .optBias(false)
                                .build())
                .add(
                        BatchNorm.builder()
                                .optEpsilon(2E-5f)
                                .optMomentum(builder.batchNormMomentum)
                                .build())
                .add(Activation.reluBlock());

        return depthWise.add(pointWise); // two blocks are merged together
    }

    /**
     * Creates a new {@link Block} of {@link MobileNetV1} with the arguments from the given {@link
     * Builder}.
     *
     * @param builder the {@link Builder} with the necessary arguments
     * @return a {@link Block} that represents the required MobileNet model
     */
    public static Block mobilenet(Builder builder) {
        // no bias in MobileNet
        SequentialBlock mobileNet = new SequentialBlock();

        mobileNet
                .add(
                        // conv1
                        new SequentialBlock()
                                .add(
                                        Conv2d.builder()
                                                .setKernelShape(new Shape(3, 3))
                                                .optBias(false)
                                                .optStride(new Shape(2, 2))
                                                .optPadding(new Shape(1, 1)) // padding = 'same'
                                                .setFilters(
                                                        (int)
                                                                (FILTERS[0]
                                                                        * builder.widthMultiplier))
                                                .build())
                                .add(
                                        BatchNorm.builder()
                                                .optEpsilon(2E-5f)
                                                .optMomentum(builder.batchNormMomentum)
                                                .build())
                                .add(Activation.reluBlock()))
                // separable conv1
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[0] * builder.widthMultiplier),
                                (int) (FILTERS[1] * builder.widthMultiplier),
                                1,
                                builder))
                // separable conv2
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[1] * builder.widthMultiplier),
                                (int) (FILTERS[2] * builder.widthMultiplier),
                                2,
                                builder))
                // separable conv3
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[2] * builder.widthMultiplier),
                                (int) (FILTERS[3] * builder.widthMultiplier),
                                1,
                                builder))
                // separable conv4
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[3] * builder.widthMultiplier),
                                (int) (FILTERS[4] * builder.widthMultiplier),
                                2,
                                builder))
                // separable conv5
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[4] * builder.widthMultiplier),
                                (int) (FILTERS[5] * builder.widthMultiplier),
                                1,
                                builder))
                // separable conv6
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[5] * builder.widthMultiplier),
                                (int) (FILTERS[6] * builder.widthMultiplier),
                                2,
                                builder))
                // separable conv7*5
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[6] * builder.widthMultiplier),
                                (int) (FILTERS[7] * builder.widthMultiplier),
                                1,
                                builder))
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[6] * builder.widthMultiplier),
                                (int) (FILTERS[7] * builder.widthMultiplier),
                                1,
                                builder))
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[6] * builder.widthMultiplier),
                                (int) (FILTERS[7] * builder.widthMultiplier),
                                1,
                                builder))
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[6] * builder.widthMultiplier),
                                (int) (FILTERS[7] * builder.widthMultiplier),
                                1,
                                builder))
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[6] * builder.widthMultiplier),
                                (int) (FILTERS[7] * builder.widthMultiplier),
                                1,
                                builder))
                // separable conv8
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[7] * builder.widthMultiplier),
                                (int) (FILTERS[8] * builder.widthMultiplier),
                                2,
                                builder))
                // separable conv9
                .add(
                        depthSeparableConv2d(
                                (int) (FILTERS[8] * builder.widthMultiplier),
                                (int) (FILTERS[9] * builder.widthMultiplier),
                                1,
                                builder)) // maybe the paper goes wrong here
                // AveragePool
                .add(Pool.globalAvgPool2dBlock())
                // FC
                .add(Linear.builder().setUnits(builder.outSize).build());

        return mobileNet;
    }

    /**
     * Creates a builder to build a {@link MobileNetV1}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new MobileNetV1.Builder();
    }

    /** The Builder to construct a {@link MobileNetV1} object. */
    public static final class Builder {

        float batchNormMomentum = 0.9f;
        float widthMultiplier = 1f; // width multiplier(also named alpha) defined in the paper
        long outSize = 10; // 10 as default for basic datasets like cifar-10 or mnist

        Builder() {}

        /**
         * Sets the widthMultiplier of MobileNet.
         *
         * @param widthMultiplier the widthMultiplier of MobileNet
         * @return this {@code Builder}
         */
        public Builder optWidthMultiplier(float widthMultiplier) {
            this.widthMultiplier = widthMultiplier;
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
         * Builds a {@link MobileNetV1} block.
         *
         * @return the {@link MobileNetV1} block
         */
        public Block build() {
            return mobilenet(this);
        }
    }
}
