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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
/**
 * {@code MobileNetV1} contains a generic implementation of Mobilenetw adapted from
 * https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenet.py (Original author weiaicunzai)
 *
 * see https://arxiv.org/pdf/1704.04861.pdf for more information about MobileNet
 */
public final class MobileNetV1 {
    private MobileNetV1(){}

    /**
     * Builds a {@link Block} that represent a depthWise-pointWise Unit used in the implementation of
     * the MobileNet Model
     *
     * @param inputChannels number of inputChannels, used for depthWise Kernel
     * @param outputChannels number of outputChannels, used for pointWise kernel
     * @param stride control the stride of depthWise Kernel
     * @param builder add the builder to obtain batchNormMomentum
     * @return a {@link Block} that represent a depthWise-pointWise Unit
     */
    public static Block DepthSeparableConv2d(int inputChannels,int outputChannels,int stride,Builder builder){
        //depthWise does not include bias
        SequentialBlock depthWise = new SequentialBlock();
        depthWise.add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3,3)) //the kernel size of depthWise is always 3
                                .optBias(false)
                                .optPadding(new Shape(1,1))     //padding = same
                                .optStride(new Shape(stride,stride))    //stride is either 2 or 1
                                .optGroups(inputChannels)           //depthWise with 1 filter per input channel
                                .setFilters(inputChannels)  //for depthWise, the input and output are the same
                                .build())
                .add(   //add a batchNorm
                        BatchNorm.builder()
                                .optEpsilon(2E-5f)
                                .optMomentum(builder.batchNormMomentum)
                                .build())
                .add(Activation.reluBlock());

        SequentialBlock pointWise = new SequentialBlock();
        pointWise.add(Conv2d.builder()
                        .setKernelShape(new Shape(1,1)) //no padding or stride
                        .setFilters(outputChannels)
                        .optBias(false)
                        .build())
                .add(
                        BatchNorm.builder()
                                .optEpsilon(2E-5f)
                                .optMomentum(builder.batchNormMomentum)
                                .build())
                .add(Activation.reluBlock());

        return depthWise.add(pointWise);    //two blocks are merged together
    }

    /**
     * Create a new {@link Block} of {@link MobileNetV1} with the arguments from the given {@link
     * Builder}.
     * @param builder the {@link Builder} with the necessary arguments
     * @return a {@link Block} that represents the required MobileNet model
     */
    public static Block mobilenet(Builder builder){
        //no bias in MobileNet
        SequentialBlock mobileNet = new SequentialBlock();


        mobileNet.add(
                        //conv1
                        new SequentialBlock().add(
                                Conv2d.builder()
                                        .setKernelShape(new Shape(3,3))
                                        .optBias(false)
                                        .optStride(new Shape(2,2))
                                        .optPadding(new Shape(1,1)) //padding = 'same'
                                        .setFilters((int) (builder.filters[0]*builder.alpha))
                                        .build()
                        ).add(
                                BatchNorm.builder()
                                        .optEpsilon(2E-5f)
                                        .optMomentum(builder.batchNormMomentum)
                                        .build()
                        ).add(Activation.reluBlock())
                )
                //separable conv1
                .add(DepthSeparableConv2d((int)(builder.filters[0]* builder.alpha)
                        ,(int)(builder.filters[1]*builder.alpha),1,builder))
                //separable conv2
                .add(DepthSeparableConv2d((int)(builder.filters[1]* builder.alpha)
                        ,(int)(builder.filters[2]* builder.alpha),2,builder))
                //separable conv3
                .add(DepthSeparableConv2d((int)(builder.filters[2]* builder.alpha)
                        ,(int)(builder.filters[3]*builder.alpha),1,builder))
                //separable conv4
                .add(DepthSeparableConv2d((int)(builder.filters[3]* builder.alpha),
                        (int)(builder.filters[4]*builder.alpha),2,builder))
                //separable conv5
                .add(DepthSeparableConv2d((int)(builder.filters[4]* builder.alpha),
                        (int)(builder.filters[5]*builder.alpha),1,builder))
                //separable conv6
                .add(DepthSeparableConv2d((int)(builder.filters[5]* builder.alpha),
                        (int)(builder.filters[6]*builder.alpha),2,builder))
                //separable conv7*5
                .add(DepthSeparableConv2d((int)(builder.filters[6]* builder.alpha),
                        (int)(builder.filters[7]*builder.alpha),1,builder))
                .add(DepthSeparableConv2d((int)(builder.filters[6]* builder.alpha),
                        (int)(builder.filters[7]*builder.alpha),1,builder))
                .add(DepthSeparableConv2d((int)(builder.filters[6]* builder.alpha),
                        (int)(builder.filters[7]*builder.alpha),1,builder))
                .add(DepthSeparableConv2d((int)(builder.filters[6]* builder.alpha),
                        (int)(builder.filters[7]*builder.alpha),1,builder))
                .add(DepthSeparableConv2d((int)(builder.filters[6]* builder.alpha),
                        (int)(builder.filters[7]*builder.alpha),1,builder))
                //separable conv8
                .add(DepthSeparableConv2d((int)(builder.filters[7]* builder.alpha),
                        (int)(builder.filters[8]*builder.alpha),2,builder))
                //separable conv9
                .add(DepthSeparableConv2d((int)(builder.filters[8]* builder.alpha),
                        (int)(builder.filters[9]*builder.alpha),1,builder)) //maybe the paper goes wrong here
                //AveragePool
                .add(Pool.globalAvgPool2dBlock())
                //FC
                .add(Linear.builder().setUnits(builder.outSize).build());

        return mobileNet;
    }
    /**
     * Creates a builder to build a {@link MobileNetV1}.
     *
     * @return a new builder
     */
    public static Builder builder(){
        return new MobileNetV1.Builder();
    }

    /**
     * The Builder to construct a {@link MobileNetV1} object.
     */
    public static final class Builder{
        float batchNormMomentum = 0.9f;
        float alpha = 1f;   //width multiplier defined in the paper

        long outSize = 10;  //10 as default for basic datasets like cifar10 or mnist

        private final int[] filters = new int[]{32,64,128,128,256,256,512,512,1024,1024};

        Builder(){}

        /**
         * Set the widthMultiplier of MobileNet
         * @param widthMultiplier the widthMultiplier of MobileNet
         * @return this {@code Builder}
         */
        public Builder setWidthMultiplier(float widthMultiplier){
            this.alpha = widthMultiplier;
            return this;
        }

        /**
         * Sets the momentum of batchNorm layer.
         *
         * @param batchNormMomentum the momentum
         * @return this {@code Builder}
         */
        public Builder setBatchNormMomentum(float batchNormMomentum){
            this.batchNormMomentum = batchNormMomentum;
            return this;
        }

        /**
         * Sets the size of the output.
         *
         * @param outSize the output size
         * @return this {@code Builder}
         */
        public Builder setOutputSize(long outSize){
            this.outSize = outSize;
            return this;
        }

        /**
         * Builds a {@link MobileNetV1} block.
         *
         * @return the {@link MobileNetV1} block
         */
        public Block build(){
            return mobilenet(this);
        }
    }


}
