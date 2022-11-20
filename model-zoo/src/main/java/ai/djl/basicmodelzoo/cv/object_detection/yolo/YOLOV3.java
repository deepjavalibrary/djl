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
package ai.djl.basicmodelzoo.cv.object_detection.yolo;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * {@code YOLOV3} contains a generic implementation of <a
 * href="https://github.com/bubbliiiing/yolo3-pytorch">yolov3</a> (Original author bubbliiiing).
 *
 * <p>Yolov3 is a fast and accurate model for ObjectDetection tasks.
 *
 * @see <a href="https://arxiv.org/abs/1804.02767">for more information and knowledge about
 *     yolov3</a>
 */
public final class YOLOV3 extends AbstractBlock {

    private static final byte VERSION = 1;

    private SequentialBlock darkNet53; // backBone of YOLOv3

    private Block lastLayer0;
    private Block layer0Output;

    private Block lastLayer1Conv;
    private Block lastLayer1UpSample;
    private Block lastLayer1;
    private Block layer1Output;

    private Block lastLayer2Conv;
    private Block lastLayer2UpSample;
    private Block lastLayer2;
    private Block layer2Output;

    static final int[] REPEATS = {
        1, 2, 8, 8, 4
    }; // the repeat times of the darkNet53 residual units

    static final int[] FILTERS = {
        32, 64, 128, 256, 512, 1024
    }; // the filters of darkNet53 residual units

    private YOLOV3(Builder builder) {
        super(VERSION);
        darkNet53 = addChildBlock("darkNet53", darkNet53(builder, true));
        lastLayer0 =
                addChildBlock(
                        "lastLayer0",
                        makeLastLayers(
                                FILTERS[4],
                                FILTERS[5],
                                builder.batchNormMomentum,
                                builder.leakyAlpha));
        layer0Output =
                addChildBlock(
                        "layer0Output",
                        makeOutputLayers(
                                FILTERS[5],
                                3 * (builder.numClasses + 5),
                                builder.batchNormMomentum,
                                builder.leakyAlpha));

        lastLayer1Conv =
                addChildBlock(
                        "lastLayer1Conv",
                        convolutionBlock(256, 1, builder.batchNormMomentum, builder.leakyAlpha));
        lastLayer1UpSample = addChildBlock("lastLayer1UpSample", upSampleBlockNearest());
        lastLayer1 =
                addChildBlock(
                        "lastLayer1",
                        makeLastLayers(
                                FILTERS[3],
                                FILTERS[4],
                                builder.batchNormMomentum,
                                builder.leakyAlpha));
        layer1Output =
                addChildBlock(
                        "layer1Output",
                        makeOutputLayers(
                                FILTERS[4],
                                3 * (builder.numClasses + 5),
                                builder.batchNormMomentum,
                                builder.leakyAlpha));

        lastLayer2Conv =
                addChildBlock(
                        "lastLayer2Conv",
                        convolutionBlock(128, 1, builder.batchNormMomentum, builder.leakyAlpha));
        lastLayer2UpSample = addChildBlock("lastLayer2UpSample", upSampleBlockNearest());
        lastLayer2 =
                addChildBlock(
                        "lastLayer2",
                        makeLastLayers(
                                FILTERS[2],
                                FILTERS[3],
                                builder.batchNormMomentum,
                                builder.leakyAlpha));
        layer2Output =
                addChildBlock(
                        "layer2Output",
                        makeOutputLayers(
                                FILTERS[3],
                                3 * (builder.numClasses + 5),
                                builder.batchNormMomentum,
                                builder.leakyAlpha));
    }

    /**
     * Builds a {@link Block} that represents an upSampleLayer(the nearest mode) for yolov3.
     *
     * @return a {@link Block} that represent an upSampleLayer for yolov3
     */
    public static Block upSampleBlockNearest() {
        // transpose + upSample + transpose
        return new SequentialBlock()
                .addSingleton(array -> array.transpose(0, 2, 3, 1))
                .addSingleton(
                        array ->
                                NDImageUtils.resize(
                                        array,
                                        (int) (array.getShape().get(1) * 2),
                                        (int) (array.getShape().get(2) * 2),
                                        Image.Interpolation.NEAREST))
                .addSingleton(array -> array.transpose(0, 3, 1, 2));
    }

    /**
     * Builds a {@link Block} that represents a conv-bn-leakyRelu unit for darkNet53.
     *
     * @param filters the number of filters for conv
     * @param kernel the kernel size for conv
     * @param batchNormMomentum the momentum for batchNorm layer
     * @param leakyAlpha the alpha for leakyRelu activation
     * @return a {@link Block} that represents a conv-bn-leakyRelu unit for darkNet53
     */
    public static Block convolutionBlock(
            int filters, int kernel, float batchNormMomentum, float leakyAlpha) {
        int pad = 0;
        if (kernel > 0) {
            pad = (kernel - 1) >> 1;
        }
        return new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setFilters(filters)
                                .setKernelShape(new Shape(kernel, kernel))
                                .optPadding(new Shape(pad, pad))
                                .build())
                .add(BatchNorm.builder().optEpsilon(2E-5f).optMomentum(batchNormMomentum).build())
                .add(Activation.leakyReluBlock(leakyAlpha));
    }

    /**
     * Builds a {@link Block} that represents the feature head in yolov3.
     *
     * @param filtersIn the number of input filters
     * @param filtersOut the number of output filters
     * @param batchNormMomentum the momentum of batchNorm layer
     * @param leakyAlpha the alpha value for leakyRelu activation
     * @return a {@link Block} that represents the feature head in yolov3.
     */
    public static Block makeLastLayers(
            int filtersIn, int filtersOut, float batchNormMomentum, float leakyAlpha) {
        return new SequentialBlock()
                .add(convolutionBlock(filtersIn, 1, batchNormMomentum, leakyAlpha))
                .add(convolutionBlock(filtersOut, 3, batchNormMomentum, leakyAlpha))
                .add(convolutionBlock(filtersIn, 1, batchNormMomentum, leakyAlpha))
                .add(convolutionBlock(filtersOut, 3, batchNormMomentum, leakyAlpha))
                .add(convolutionBlock(filtersIn, 1, batchNormMomentum, leakyAlpha));
    }

    /**
     * Builds a {@link Block} that represents the output layer of yolov3.
     *
     * @param filtersOut the number of output filters
     * @param outClass the number of output classes
     * @param batchNormMomentum the momentum for batchNorm layer
     * @param leakyAlpha the alpha for leakyRelu activation
     * @return a {@link Block} that represents the output layer of yolov3.
     */
    public static Block makeOutputLayers(
            int filtersOut, int outClass, float batchNormMomentum, float leakyAlpha) {
        return new SequentialBlock()
                .add(convolutionBlock(filtersOut, 3, batchNormMomentum, leakyAlpha))
                .add(Conv2d.builder().setFilters(outClass).setKernelShape(new Shape(1, 1)).build());
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList xList = darkNet53.forward(parameterStore, inputs, training);
        NDArray x0 = xList.get(7);
        NDArray x1 = xList.get(6);
        NDArray x2 = xList.get(5);

        // the first feature layer
        NDList out0Branch = lastLayer0.forward(parameterStore, new NDList(x0), training);
        NDList out0 = layer0Output.forward(parameterStore, out0Branch, training);
        NDList x1In = lastLayer1Conv.forward(parameterStore, out0Branch, training);
        x1In = lastLayer1UpSample.forward(parameterStore, x1In, training);
        x1In = new NDList(x1In.singletonOrThrow().concat(x1, 1));

        NDList out1Branch = lastLayer1.forward(parameterStore, x1In, training);
        NDList out1 = layer1Output.forward(parameterStore, out1Branch, training);

        NDList x2In = lastLayer2Conv.forward(parameterStore, out1Branch, training);
        x2In = lastLayer2UpSample.forward(parameterStore, x2In, training);
        x2In = new NDList(x2In.singletonOrThrow().concat(x2, 1));

        // the third feature layer
        NDList out2 = lastLayer2.forward(parameterStore, x2In, training);
        out2 = layer2Output.forward(parameterStore, out2, training);

        // Outputs
        return new NDList(
                out0.singletonOrThrow(), out1.singletonOrThrow(), out2.singletonOrThrow());
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape[] current = inputShapes;
        Shape[] outputs = new Shape[3];
        Shape[] darkNetOutputs = new Shape[8];
        int index = 0;
        for (String name : children.keys()) {
            Block block = children.get(name);
            if (name.contains("darkNet")) {
                darkNetOutputs = block.getOutputShapes(current);
                current = new Shape[] {darkNetOutputs[7]};
            } else if (name.contains("lastLayer")) {
                if ("05lastLayer1UpSample".equals(name)) {
                    current = block.getOutputShapes(current);
                    current =
                            new Shape[] {
                                new Shape(
                                        current[0].get(0),
                                        current[0].get(1) + darkNetOutputs[6].get(1),
                                        current[0].get(2),
                                        current[0].get(3))
                            };
                } else if ("09lastLayer2UpSample".equals(name)) {
                    current = block.getOutputShapes(current);
                    current =
                            new Shape[] {
                                new Shape(
                                        current[0].get(0),
                                        current[0].get(1) + darkNetOutputs[5].get(1),
                                        current[0].get(2),
                                        current[0].get(3))
                            };
                } else {
                    current = block.getOutputShapes(current);
                }
            } else if (!name.contains("Output")) {
                current = block.getOutputShapes(current);
            } else { // name.contains("Output")
                Shape[] output = block.getOutputShapes(current);
                outputs[index++] = output[0];
            }
        }
        return outputs;
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] current = inputShapes;
        Shape[] darkNetOutputs = new Shape[8];
        for (String name : children.keys()) {
            Block block = children.get(name);
            block.initialize(manager, dataType, current);
            if (name.contains("darkNet")) {
                darkNetOutputs = block.getOutputShapes(current);
                current = new Shape[] {darkNetOutputs[7]};
            } else if (name.contains("lastLayer")) {
                if ("05lastLayer1UpSample".equals(name)) {
                    current = block.getOutputShapes(current);
                    current =
                            new Shape[] {
                                new Shape(
                                        current[0].get(0),
                                        current[0].get(1) + darkNetOutputs[6].get(1),
                                        current[0].get(2),
                                        current[0].get(3))
                            };
                } else if ("09lastLayer2UpSample".equals(name)) {
                    current = block.getOutputShapes(current);
                    current =
                            new Shape[] {
                                new Shape(
                                        current[0].get(0),
                                        current[0].get(1) + darkNetOutputs[5].get(1),
                                        current[0].get(2),
                                        current[0].get(3))
                            };
                } else {
                    current = block.getOutputShapes(current);
                }
            } else if (!name.contains("Output")) {
                current = block.getOutputShapes(current);
            } else { // name.contains("Output")
                block.getOutputShapes(current);
            }
        }
    }

    /**
     * Builds a {@link Block} that a basic residual block unit used in DarkNet53.
     *
     * @param filters the output filter of the Convolutional Layer
     * @param batchNormMomentum the momentum used for computing batchNorm
     * @param leakyAlpha the alpha used in LeakyRelu Function
     * @return a basic residual block unit used in DarkNet53
     */
    public static Block basicBlock(int filters, float batchNormMomentum, float leakyAlpha) {
        SequentialBlock block = new SequentialBlock();
        block.add(Conv2d.builder().setFilters(filters / 2).setKernelShape(new Shape(1, 1)).build())
                .add(BatchNorm.builder().optEpsilon(2E-5f).optMomentum(batchNormMomentum).build())
                .add(Activation.leakyReluBlock(leakyAlpha))
                .add(
                        Conv2d.builder()
                                .setFilters(filters)
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .build())
                .add(BatchNorm.builder().optEpsilon(2E-5f).optMomentum(batchNormMomentum).build())
                .add(Activation.leakyReluBlock(leakyAlpha));
        return new ParallelBlock(
                list ->
                        new NDList(
                                NDArrays.add(
                                        list.get(0).singletonOrThrow(),
                                        list.get(1).singletonOrThrow())),
                Arrays.asList(block, Blocks.identityBlock()));
    }

    /**
     * Creates repeated Residual Blocks used in DarkNet53.
     *
     * @param filters the output filters of the final Convolutional Layer
     * @param repeats the repeat times of a residual unit
     * @param batchNormMomentum the momentum used for computing batchNorm
     * @param leakyAlpha the alpha used in LeakyRelu Function
     * @return several repeats of a residual block
     */
    public static Block makeLayer(
            int filters, int repeats, float batchNormMomentum, float leakyAlpha) {
        List<Block> layer = new ArrayList<>();
        SequentialBlock convolutionalLayer = new SequentialBlock();
        convolutionalLayer
                .add(
                        Conv2d.builder()
                                .setFilters(filters)
                                .setKernelShape(new Shape(3, 3))
                                .optStride(new Shape(2, 2))
                                .optPadding(new Shape(1, 1))
                                .build())
                .add(BatchNorm.builder().optEpsilon(2E-5f).optMomentum(batchNormMomentum).build())
                .add(Activation.leakyReluBlock(leakyAlpha));
        for (int i = 0; i < repeats; i++) {
            layer.add(basicBlock(filters, batchNormMomentum, leakyAlpha));
        }
        return new SequentialBlock().add(convolutionalLayer).addAll(layer);
    }

    private static SequentialBlock darkNet53(Builder builder, boolean setReturnIntermediate) {
        SequentialBlock darkNet53 = new SequentialBlock();
        darkNet53.setReturnIntermediate(setReturnIntermediate); // return interMediate results;
        darkNet53
                .add(
                        Conv2d.builder()
                                .setFilters(FILTERS[0])
                                .optPadding(new Shape(1, 1))
                                .setKernelShape(new Shape(3, 3))
                                .build())
                .add(
                        BatchNorm.builder()
                                .optEpsilon(2E-5f)
                                .optMomentum(builder.batchNormMomentum)
                                .build())
                .add(Activation.leakyReluBlock(builder.leakyAlpha))
                .add(
                        makeLayer(
                                FILTERS[1],
                                REPEATS[0],
                                builder.batchNormMomentum,
                                builder.leakyAlpha))
                .add(
                        makeLayer(
                                FILTERS[2],
                                REPEATS[1],
                                builder.batchNormMomentum,
                                builder.leakyAlpha))
                .add(
                        makeLayer(
                                FILTERS[3],
                                REPEATS[2],
                                builder.batchNormMomentum,
                                builder.leakyAlpha))
                .add(
                        makeLayer(
                                FILTERS[4],
                                REPEATS[3],
                                builder.batchNormMomentum,
                                builder.leakyAlpha))
                .add(
                        makeLayer(
                                FILTERS[5],
                                REPEATS[4],
                                builder.batchNormMomentum,
                                builder.leakyAlpha));
        return darkNet53;
    }

    /**
     * Creates a builder to build a {@link YOLOV3}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link YOLOV3} object. */
    public static final class Builder {
        int numClasses = 20;
        float batchNormMomentum = 0.9f;
        float leakyAlpha = 0.1f;
        int darkNetOutSize = 10;

        /**
         * Sets the number of classes for yolov3.
         *
         * @param numClasses the number of classes
         * @return this {@code Builder}
         */
        public Builder setNumClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        /**
         * Sets the momentum for batchNorm layer.
         *
         * @param batchNormMomentum the momentum for batchNorm layer
         * @return this {@code Builder}
         */
        public Builder optBatchNormMomentum(float batchNormMomentum) {
            this.batchNormMomentum = batchNormMomentum;
            return this;
        }

        /**
         * Sets the alpha for leakyRelu activation.
         *
         * @param leakyAlpha the alpha for leakyRelu activation
         * @return this {@code Builder}
         */
        public Builder optLeakyAlpha(float leakyAlpha) {
            this.leakyAlpha = leakyAlpha;
            return this;
        }

        /**
         * Sets the out size of darkNet for testing.
         *
         * @param darkNetOutSize the out size of darkNet
         * @return this {@code Builder}
         */
        public Builder optDarkNetOutSize(int darkNetOutSize) {
            this.darkNetOutSize = darkNetOutSize;
            return this;
        }

        /**
         * Builds a {@link YOLOV3} block.
         *
         * @return a {@link YOLOV3} block
         */
        public Block build() {
            return new YOLOV3(this);
        }

        /**
         * Builds a {@link Block} that represents the backbone of yolov3, which is called DarkNet53.
         * This can be used for testing and transfer learning.
         *
         * @return a {@link Block} that represents darkNet53
         */
        public Block buildDarkNet() {
            Block block = darkNet53(this, false);
            return new SequentialBlock()
                    .add(block)
                    .add(Pool.globalAvgPool2dBlock())
                    .add(Linear.builder().setUnits(darkNetOutSize).build());
        }
    }
}
