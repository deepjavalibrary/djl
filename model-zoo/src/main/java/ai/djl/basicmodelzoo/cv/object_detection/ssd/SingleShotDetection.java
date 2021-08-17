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
package ai.djl.basicmodelzoo.cv.object_detection.ssd;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.MultiBoxPrior;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code SingleShotDetection} is an implementation of {@link Block} that implements a Single Shot
 * Detection (SSD) model for object detection.
 */
public final class SingleShotDetection extends AbstractBlock {

    private static final byte VERSION = 2;

    private List<Block> features;
    private List<Block> classPredictionBlocks;
    private List<Block> anchorPredictionBlocks;

    private List<MultiBoxPrior> multiBoxPriors;
    private int numClasses;

    private SingleShotDetection(Builder builder) {
        super(VERSION);
        features = builder.features;
        features.forEach((block) -> addChildBlock(block.getClass().getSimpleName(), block));
        numClasses = builder.numClasses;
        classPredictionBlocks = builder.classPredictionBlocks;
        classPredictionBlocks.forEach(
                (block) -> addChildBlock(block.getClass().getSimpleName(), block));
        anchorPredictionBlocks = builder.anchorPredictionBlocks;
        anchorPredictionBlocks.forEach(
                (block) -> addChildBlock(block.getClass().getSimpleName(), block));
        multiBoxPriors = builder.multiBoxPriors;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList networkOutput = inputs;
        NDArray[] anchorsOutputs = new NDArray[features.size()];
        NDArray[] classOutputs = new NDArray[features.size()];
        NDArray[] boundingBoxOutputs = new NDArray[features.size()];
        for (int i = 0; i < features.size(); i++) {
            networkOutput = features.get(i).forward(parameterStore, networkOutput, training);

            MultiBoxPrior multiBoxPrior = multiBoxPriors.get(i);

            anchorsOutputs[i] = multiBoxPrior.generateAnchorBoxes(networkOutput.singletonOrThrow());
            classOutputs[i] =
                    classPredictionBlocks
                            .get(i)
                            .forward(parameterStore, networkOutput, training)
                            .singletonOrThrow();
            boundingBoxOutputs[i] =
                    anchorPredictionBlocks
                            .get(i)
                            .forward(parameterStore, networkOutput, training)
                            .singletonOrThrow();
        }
        NDArray anchors = NDArrays.concat(new NDList(anchorsOutputs), 1);
        NDArray classPredictions = concatPredictions(new NDList(classOutputs));
        NDArray boundingBoxPredictions = concatPredictions(new NDList(boundingBoxOutputs));
        classPredictions = classPredictions.reshape(classPredictions.size(0), -1, numClasses + 1);

        return new NDList(anchors, classPredictions, boundingBoxPredictions);
    }

    private NDArray concatPredictions(NDList output) {
        // transpose and batch flatten
        NDArray[] flattenOutput =
                output.stream()
                        .map(array -> array.transpose(0, 2, 3, 1).reshape(array.size(0), -1))
                        .toArray(NDArray[]::new);
        return NDArrays.concat(new NDList(flattenOutput), 1);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        try (NDManager manager = NDManager.newBaseManager()) {
            // TODO: output shape is wrong
            Shape[] childInputShapes = inputShapes;
            Shape[] anchorShapes = new Shape[features.size()];
            Shape[] classPredictionShapes = new Shape[features.size()];
            Shape[] anchorPredictionShapes = new Shape[features.size()];
            for (int i = 0; i < features.size(); i++) {
                childInputShapes = features.get(i).getOutputShapes(childInputShapes);
                anchorShapes[i] =
                        multiBoxPriors
                                .get(i)
                                .generateAnchorBoxes(manager.ones(childInputShapes[0]))
                                .getShape();
                classPredictionShapes[i] =
                        classPredictionBlocks.get(i).getOutputShapes(childInputShapes)[0];
                anchorPredictionShapes[i] =
                        anchorPredictionBlocks.get(i).getOutputShapes(childInputShapes)[0];
            }
            Shape anchorOutputShape = new Shape();
            for (Shape shape : anchorShapes) {
                anchorOutputShape = concatShape(anchorOutputShape, shape, 1);
            }

            NDList classPredictions = new NDList();
            for (Shape shape : classPredictionShapes) {
                classPredictions.add(manager.ones(shape));
            }
            NDArray classPredictionOutput = concatPredictions(classPredictions);
            Shape classPredictionOutputShape =
                    classPredictionOutput
                            .reshape(classPredictionOutput.size(0), -1, numClasses + 1)
                            .getShape();
            NDList anchorPredictions = new NDList();
            for (Shape shape : anchorPredictionShapes) {
                anchorPredictions.add(manager.ones(shape));
            }
            Shape anchorPredictionOutputShape = concatPredictions(anchorPredictions).getShape();
            return new Shape[] {
                anchorOutputShape, classPredictionOutputShape, anchorPredictionOutputShape
            };
        }
    }

    private Shape concatShape(Shape shape, Shape concat, int axis) {
        if (shape.dimension() == 0) {
            return concat;
        }
        if (shape.dimension() != concat.dimension()) {
            throw new IllegalArgumentException("Shapes must have same dimensions");
        }
        long[] dimensions = new long[shape.dimension()];
        for (int i = 0; i < shape.dimension(); i++) {
            if (axis == i) {
                dimensions[i] = shape.get(i) + concat.get(i);
            } else {
                if (shape.get(i) != concat.get(i)) {
                    throw new UnsupportedOperationException(
                            "These shapes cannot be concatenated along axis " + i);
                }
                dimensions[i] = shape.get(i);
            }
        }
        return new Shape(dimensions);
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        Shape[] shapes = inputShapes;
        for (int i = 0; i < features.size(); i++) {
            features.get(i).initialize(manager, dataType, shapes);
            shapes = features.get(i).getOutputShapes(shapes);
            classPredictionBlocks.get(i).initialize(manager, dataType, shapes);
            anchorPredictionBlocks.get(i).initialize(manager, dataType, shapes);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == version) {
            readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }

    /**
     * Creates a {@link Block} that reduces the size of a convolutional block by half.
     *
     * @param numFilters the number of filters
     * @return a {@link Block} that reduces the size of a convolutional block by half
     */
    public static SequentialBlock getDownSamplingBlock(int numFilters) {
        SequentialBlock sequentialBlock = new SequentialBlock();
        for (int i = 0; i < 2; i++) {
            sequentialBlock
                    .add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(3, 3))
                                    .setFilters(numFilters)
                                    .optPadding(new Shape(1, 1))
                                    .build())
                    .add(BatchNorm.builder().build())
                    .add(Activation::relu);
        }
        sequentialBlock.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2), new Shape(0, 0)));
        return sequentialBlock;
    }

    /**
     * Creates a class prediction block used in an SSD.
     *
     * @param numAnchors the number of anchors
     * @param numClasses the number of classes
     * @return a class prediction block used in an SSD
     */
    public static Conv2d getClassPredictionBlock(int numAnchors, int numClasses) {
        return Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters((numClasses + 1) * numAnchors)
                .optPadding(new Shape(1, 1))
                .build();
    }

    /**
     * Creates a anchor prediction block used in an SSD.
     *
     * @param numAnchors the number of anchors
     * @return a anchor prediction block used in an SSD
     */
    public static Conv2d getAnchorPredictionBlock(int numAnchors) {
        return Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(4 * numAnchors)
                .optPadding(new Shape(1, 1))
                .build();
    }

    /**
     * Creates a builder to build a {@link SingleShotDetection}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link SingleShotDetection}. */
    public static class Builder {

        private Block network;
        private int numFeatures = -1;
        private List<Block> features;
        private List<List<Float>> sizes;
        private List<List<Float>> ratios;
        private List<Block> classPredictionBlocks = new ArrayList<>();
        private List<Block> anchorPredictionBlocks = new ArrayList<>();
        private List<MultiBoxPrior> multiBoxPriors = new ArrayList<>();
        private int numClasses;
        private boolean globalPool = true;

        Builder() {}

        /**
         * Sets the list of sizes of generated anchor boxes.
         *
         * @param sizes size of the input
         * @return Returns this Builder
         */
        public Builder setSizes(List<List<Float>> sizes) {
            this.sizes = sizes;
            return this;
        }

        /**
         * Sets the list of aspect ratios of generated anchor boxes.
         *
         * @param ratios size of the input
         * @return Returns this Builder
         */
        public Builder setRatios(List<List<Float>> ratios) {
            this.ratios = ratios;
            return this;
        }

        /**
         * Sets the number of classes of objects to be detected.
         *
         * @param numClasses number of classes
         * @return Returns this Builder
         */
        public Builder setNumClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        /**
         * Sets the base network for the SSD framework.
         *
         * @param network Base network
         * @return Returns this Builder
         */
        public Builder setBaseNetwork(Block network) {
            this.network = network;
            return this;
        }

        /**
         * Sets the number of down sampling blocks to be applied. Down-sampling blocks are applied
         * to the base network successively, and feature maps are drawn from the each of the blocks.
         * This value is ignored if features is also set.
         *
         * @param numFeatures Number of down sampling blocks to be applied
         * @return Returns this Builder
         */
        public Builder setNumFeatures(int numFeatures) {
            this.numFeatures = numFeatures;
            return this;
        }

        /**
         * Sets the {@code Conv2d} blocks to be appended to the network to get multi-output network.
         *
         * @param features List of {@code Conv2d} blocks to be appended
         * @return Returns this Builder
         */
        public Builder optFeatures(List<Block> features) {
            this.features = features;
            return this;
        }

        /**
         * Sets the boolean whether to attach a global average pooling layer as the last output
         * layer.
         *
         * @param globalPool Whether to attach a global average pooling layer as the last output
         *     layer
         * @return Returns this Builder
         */
        public Builder optGlobalPool(boolean globalPool) {
            this.globalPool = globalPool;
            return this;
        }

        /**
         * Builds a {@link SingleShotDetection} block.
         *
         * @return the {@link SingleShotDetection} block
         */
        public SingleShotDetection build() {
            if (features == null && numFeatures < 0) {
                throw new IllegalArgumentException("Either numFeatures or features must be set");
            } else if (features == null) {
                features = new ArrayList<>();
                features.add(network);
                for (int i = 0; i < numFeatures; i++) {
                    features.add(getDownSamplingBlock(128));
                }
            }
            if (globalPool) {
                features.add(
                        LambdaBlock.singleton(
                                array -> {
                                    NDArray result = Pool.globalAvgPool2d(array);
                                    // result shape: (N, C) MXNet multi-box takes (N, C, 1, 1)
                                    return result.reshape(result.getShape().add(1, 1));
                                }));
            }
            int numberOfFeatureMaps = features.size();
            if (sizes.size() != ratios.size() || sizes.size() != numberOfFeatureMaps) {
                throw new IllegalArgumentException(
                        "Sizes and ratios must be of size: " + numberOfFeatureMaps);
            }
            for (int i = 0; i < numberOfFeatureMaps; i++) {
                List<Float> size = sizes.get(i);
                List<Float> ratio = ratios.get(i);

                int numAnchors = size.size() + ratio.size() - 1;
                classPredictionBlocks.add(getClassPredictionBlock(numAnchors, numClasses));
                anchorPredictionBlocks.add(getAnchorPredictionBlock(numAnchors));
                multiBoxPriors.add(MultiBoxPrior.builder().setSizes(size).setRatios(ratio).build());
            }
            return new SingleShotDetection(this);
        }
    }
}
