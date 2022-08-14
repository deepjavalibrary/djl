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

import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * {@code MobileNetV2} contains a generic implementation of MobilenetV2 adapted from
 * https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenetv2.py (Original
 * author weiaicunzai).
 *
 * <p>see https://arxiv.org/pdf/1801.04381.pdf for more information about MobileNetV2
 */
public final class MobileNetV2 {

    public static final int FILTERLENGTH = 9;
    public static final int REPEATLENGTH = 9;
    public static final int STRIDELENGTH = 9;
    public static final int MULTILENGTH = 7;

    private MobileNetV2() {}
    /**
     * Builds a {@link Block} that represent an inverted residual Unit used in the implementation of
     * the MobileNetV2 Model.
     *
     * @param inputChannels number of inputChannels of the block
     * @param outputChannels number of outputChannels of the block
     * @param stride control the stride of a depthWise kernel
     * @param t the multiTime of the first pointWise Block
     * @param batchNormMomentum the momentum of batchNormLayer
     * @return a {@link Block} that represent an inverted residual Unit
     */
    public static Block linearBottleNeck(
            int inputChannels, int outputChannels, int stride, int t, float batchNormMomentum) {
        SequentialBlock block = new SequentialBlock();
        block.add(
                        Conv2d.builder() // PointWise
                                .setFilters(inputChannels * t)
                                .setKernelShape(new Shape(1, 1))
                                .optBias(false)
                                .build())
                // add a batchNorm
                .add(BatchNorm.builder().optEpsilon(2E-5f).optMomentum(batchNormMomentum).build())
                // add a relu
                .add(Activation.relu6Block())
                .add(
                        Conv2d.builder() // DepthWise
                                .setKernelShape(new Shape(3, 3))
                                .setFilters(inputChannels * t)
                                .optStride(new Shape(stride, stride))
                                .optPadding(new Shape(1, 1))
                                .optGroups(inputChannels * t)
                                .optBias(false)
                                .build())
                .add(BatchNorm.builder().optEpsilon(2E-5f).optMomentum(batchNormMomentum).build())
                .add(Activation.relu6Block())
                .add(
                        Conv2d.builder() // PointWise
                                .setFilters(outputChannels)
                                .setKernelShape(new Shape(1, 1))
                                .optBias(false)
                                .build())
                .add(BatchNorm.builder().optEpsilon(2E-5f).optMomentum(batchNormMomentum).build());

        // if dimMatch,then add x
        if (stride == 1 && inputChannels == outputChannels) {
            return new ParallelBlock(
                    list ->
                            new NDList(
                                    NDArrays.add(
                                            list.get(0).singletonOrThrow(),
                                            list.get(1).singletonOrThrow())),
                    Arrays.asList(block, Blocks.identityBlock()));
        }
        return block; // No relu6Block here
    }

    /**
     * Builds a {@link Block} that represent multiple repeats of an inverted residual Unit.
     *
     * @param repeat the repeatTimes of an inverted residual Block
     * @param inputChannels number of inputChannels of the block
     * @param outputChannels number of outputChannels of the block
     * @param stride the stride of an inverted residual Unit
     * @param t the multipleTime of a pointWise Kernel
     * @param batchNormMomentum the momentum of batchNormLayer
     * @return a {@link Block} that represent several repeated inverted residual Units
     */
    public static Block makeStage(
            int repeat,
            int inputChannels,
            int outputChannels,
            int stride,
            int t,
            float batchNormMomentum) {
        SequentialBlock layers = new SequentialBlock();
        layers.add(linearBottleNeck(inputChannels, outputChannels, stride, t, batchNormMomentum));
        for (int i = 0; i < repeat - 1; i++) {
            layers.add(linearBottleNeck(outputChannels, outputChannels, 1, t, batchNormMomentum));
        }
        return layers;
    }

    /**
     * Creates a new {@link Block} of {@link MobileNetV2} with the arguments from the given {@link
     * Builder}.
     *
     * @param builder the {@link Builder} with the necessary arguments
     * @return a {@link Block} that represents the required MobileNetV2 model
     */
    public static Block mobilenetV2(Builder builder) {
        SequentialBlock mobileNet = new SequentialBlock();
        SequentialBlock pre = new SequentialBlock();
        for (int i = 0; i < builder.repeatTimes[0]; i++) { // add as a sequence
            pre.add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(1, 1))
                                    .setFilters(builder.filters[0])
                                    .optStride(new Shape(builder.strides[0], builder.strides[0]))
                                    .optBias(false)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(2E-5f)
                                    .optMomentum(builder.batchNormMomentum)
                                    .build())
                    .add(Activation.relu6Block());
        }

        List<Block> bottleNecks = new ArrayList<>();
        for (int i = 0; i < MULTILENGTH; i++) {
            bottleNecks.add(
                    makeStage(
                            builder.repeatTimes[i + 1],
                            builder.filters[i],
                            builder.filters[i + 1],
                            builder.strides[i + 1],
                            builder.multiTimes[i],
                            builder.batchNormMomentum));
        }

        SequentialBlock conv1 = new SequentialBlock();
        for (int i = 0; i < builder.repeatTimes[8]; i++) {
            conv1.add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(1, 1))
                                    .setFilters(builder.filters[8])
                                    .optStride(new Shape(builder.strides[8], builder.strides[8]))
                                    .optBias(false)
                                    .build())
                    .add(
                            BatchNorm.builder()
                                    .optEpsilon(2E-5f)
                                    .optMomentum(builder.batchNormMomentum)
                                    .build())
                    .add(Activation.relu6Block());
        }

        Block conv2 =
                Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters((int) builder.outSize)
                        .build();

        return mobileNet
                .add(pre)
                .addAll(bottleNecks)
                .add(conv1)
                .add(Pool.globalAvgPool2dBlock())
                .addSingleton(
                        array -> array.reshape(array.getShape().get(0), builder.filters[8], 1, 1))
                // reshape for conv1*1
                .add(conv2)
                // reshape for output
                .addSingleton(array -> array.reshape(array.getShape().get(0), builder.outSize));
    }

    /**
     * Creates a builder to build a {@link MobileNetV2}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link MobileNetV2} object. */
    public static final class Builder {

        float batchNormMomentum = 0.9f;
        long outSize = 10; // 10 as default for basic datasets like cifar-10 or mnist
        int[] repeatTimes = {
            1, 1, 2, 3, 4, 3, 3, 1, 1
        }; // repeatTimes(n) of each block defined in the paper
        int[] filters = {
            32, 16, 24, 32, 64, 96, 160, 320, 1280
        }; // filters(c) of each Block defined in the paper
        int[] strides = {
            2, 1, 2, 2, 2, 1, 2, 1, 1
        }; // strides(s) of each block defined in the paper
        int[] multiTimes = {
            1, 6, 6, 6, 6, 6, 6
        }; // multipleTimes(t) of each linearBottleneck defined in the paper

        Builder() {}

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
         * Sets the filters(the value c defined in the paper) of customized MobileNetV2.
         *
         * @param filters the customized filter
         * @return this {@code Builder}
         */
        public Builder optFilters(int[] filters) {
            if (filters.length != FILTERLENGTH) {
                throw new IllegalArgumentException(
                        String.format(
                                "optFilters requires filters of length %d, but was given filters of"
                                        + " length %d instead",
                                FILTERLENGTH, filters.length));
            }
            this.filters = filters;
            return this;
        }

        /**
         * Sets the repeatTimes(the value n defined in the paper) of each block of MobileNetV2.
         *
         * @param repeatTimes the customized repeatTimes
         * @return this {@code Builder}
         */
        public Builder optRepeatTimes(int[] repeatTimes) {
            if (repeatTimes.length != REPEATLENGTH) {
                throw new IllegalArgumentException(
                        String.format(
                                "optRepeatTimes requires repeatTimes of length %d, but was given"
                                        + " repeatTimes of length %d instead",
                                REPEATLENGTH, repeatTimes.length));
            }
            this.repeatTimes = repeatTimes;
            return this;
        }

        /**
         * Sets the strides(the value s defined in the paper) of each block of MobileNetV2.
         *
         * @param strides the customized strides
         * @return this {@code Builder}
         */
        public Builder optStrides(int[] strides) {
            if (strides.length != STRIDELENGTH) {
                throw new IllegalArgumentException(
                        String.format(
                                "optStrides requires strides of length %d, but was given strides of"
                                        + " length %d instead",
                                STRIDELENGTH, strides.length));
            }
            this.strides = strides;
            return this;
        }

        /**
         * Sets the multiTimes(the value t defined in the paper) of each bottleNeck of MobileNetV2.
         *
         * @param multiTimes the customized multiTimes
         * @return this {@code Builder}
         */
        public Builder optMultiTimes(int[] multiTimes) {
            if (multiTimes.length != MULTILENGTH) {
                throw new IllegalArgumentException(
                        String.format(
                                "optMultiTimes requires multiTimes of length %d, but was given"
                                        + " multiTimes of length %d instead",
                                MULTILENGTH, multiTimes.length));
            }
            this.multiTimes = multiTimes;
            return this;
        }

        /**
         * Builds a {@link MobileNetV2} block.
         *
         * @return the {@link MobileNetV2} block
         */
        public Block build() {
            return mobilenetV2(this);
        }
    }
}
