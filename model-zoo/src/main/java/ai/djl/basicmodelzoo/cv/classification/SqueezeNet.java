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

import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import java.util.Arrays;

/**
 * {@code SqueezeNet} contains a generic implementation of Squeezenet adapted from [torchvision
 * implmentation](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py)
 *
 * <p>Squeezenet is an efficient NN used for Image classification. It provides both performance
 * boost and a tiny size. It's a good choice to adopt squeezenet for application runs on Mobile or
 * Edge devices. Implementing the original squeezenet from Forrest N. Iandola, Song Han, Matthew W.
 * Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer. "SQUEEZENET: ALEXNET-LEVEL ACCURACY
 * WITH 50X FEWER PARAMETERS AND 0.5MB MODEL SIZE"
 */
public final class SqueezeNet {

    private SqueezeNet() {}

    static Block fire(int squeezePlanes, int expand1x1Planes, int expand3x3Planes) {

        SequentialBlock squeezeWithActivation =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setFilters(squeezePlanes)
                                        .setKernelShape(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu);
        SequentialBlock expand1x1 =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setFilters(expand1x1Planes)
                                        .setKernelShape(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu);
        SequentialBlock expand3x3 =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setFilters(expand3x3Planes)
                                        .setKernelShape(new Shape(3, 3))
                                        .optPadding(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu);

        return new SequentialBlock()
                .add(squeezeWithActivation)
                .add(
                        new ParallelBlock(
                                list ->
                                        new NDList(
                                                NDArrays.concat(
                                                        list.get(0).addAll(list.get(1)), 1)),
                                Arrays.asList(expand1x1, expand3x3)));
    }

    /**
     * Construct squeezenet v1.1.
     *
     * @param outSize the number of output classes
     * @return squeezenet {@link Block}
     */
    public static Block squeezenet(int outSize) {
        return new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setFilters(64)
                                .setKernelShape(new Shape(3, 3))
                                .optStride(new Shape(2, 2))
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(0, 0), true))
                .add(fire(16, 64, 64))
                .add(fire(16, 64, 64))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(0, 0), true))
                .add(fire(32, 128, 128))
                .add(fire(32, 128, 128))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(0, 0), true))
                .add(fire(48, 192, 192))
                .add(fire(48, 192, 192))
                .add(fire(64, 256, 256))
                .add(fire(64, 256, 256))
                // Classifier
                .add(Dropout.builder().optRate(0.5f).build())
                .add(Conv2d.builder().setFilters(outSize).setKernelShape(new Shape(1, 1)).build())
                .add(Activation::relu)
                .add(Pool.globalAvgPool2dBlock())
                .add(Blocks.batchFlattenBlock());
    }
}
