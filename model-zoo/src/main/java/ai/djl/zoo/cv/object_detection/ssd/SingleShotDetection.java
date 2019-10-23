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
package ai.djl.zoo.cv.object_detection.ssd;

import ai.djl.modality.cv.MultiBoxPrior;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2D;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.zoo.cv.classification.ResNetV1;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class SingleShotDetection extends ParameterBlock {
    private static final byte VERSION = 1;
    private List<Block> features;
    private List<Block> classPredictionBlocks;
    private List<Block> anchorPredictionBlocks;

    private List<List<Float>> sizes;
    private List<List<Float>> ratios;
    private int numClasses;
    private boolean useBatchNorm;
    private NDArray anchorBoxes;

    private SingleShotDetection(Builder builder) {
        this.features = builder.features;
        this.sizes = builder.sizes;
        this.ratios = builder.ratios;
        this.numClasses = builder.numClasses;
        this.useBatchNorm = builder.useBatchNorm;
        classPredictionBlocks = builder.classPredictionBlocks;
        anchorPredictionBlocks = builder.anchorPredictionBlocks;
    }

    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        NDList networkOutput = inputs;
        NDArray classOutput = null;
        NDArray anchorOutput = null;
        for (int i = 0; i < features.size(); i++) {
            networkOutput = features.get(i).forward(parameterStore, networkOutput);

            MultiBoxPrior multiBoxPriors =
                    new MultiBoxPrior.Builder()
                            .setSizes(sizes.get(i))
                            .setRatios(ratios.get(i))
                            .build();
            if (i == 0) {
                anchorBoxes = multiBoxPriors.generateAnchorBoxes(networkOutput.head());
            } else {
                anchorBoxes =
                        anchorBoxes.concat(
                                multiBoxPriors.generateAnchorBoxes(networkOutput.head()), 1);
            }
            Block classPredictionBlock = classPredictionBlocks.get(i);
            NDArray classPrediction =
                    classPredictionBlock.forward(parameterStore, networkOutput).head();
            classPrediction = classPrediction.transpose(0, 2, 3, 1);
            Block anchorPredictionBlock = anchorPredictionBlocks.get(i);
            NDArray anchorPrediction =
                    anchorPredictionBlock.forward(parameterStore, networkOutput).head();
            anchorPrediction = anchorPrediction.transpose(0, 2, 3, 1);
            if (i == 0) {
                classOutput =
                        classPrediction.reshape(classPredictionShape(classPrediction.getShape()));
                anchorOutput =
                        anchorPrediction.reshape(
                                anchorPredictionShape(anchorPrediction.getShape()));
            } else {
                classOutput =
                        classOutput.concat(
                                classPrediction.reshape(
                                        classPredictionShape(classPrediction.getShape())),
                                1);
                anchorOutput =
                        anchorOutput.concat(
                                anchorPrediction.reshape(
                                        anchorPredictionShape(anchorPrediction.getShape())),
                                1);
            }
            if (useBatchNorm) {
                Block batchNorm = new BatchNorm.Builder().optEpsilon(2E-5f).build();
                networkOutput = batchNorm.forward(parameterStore, networkOutput);
            }
            networkOutput = Activation.reluBlock().forward(parameterStore, networkOutput);
        }
        return new NDList(classOutput, anchorOutput);
    }

    public NDArray getAnchorBoxes() {
        return anchorBoxes;
    }

    private Shape anchorPredictionShape(Shape shape) {
        long batchSize = shape.get(0);
        long batchVolume = 1;
        for (int i = 0; i < shape.dimension(); i++) {
            batchVolume = batchVolume * shape.get(i);
        }
        batchVolume = batchVolume / batchSize;
        return new Shape(batchSize, batchVolume);
    }

    private Shape classPredictionShape(Shape shape) {
        long batchSize = shape.get(0);
        long batchVolume = 1;
        for (int i = 0; i < shape.dimension(); i++) {
            batchVolume = batchVolume * shape.get(i);
        }
        batchVolume = batchVolume / batchSize;

        return new Shape(batchSize, batchVolume / (numClasses + 1), numClasses + 1);
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("SSDBlock has no parameters");
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        Shape[] childInputShapes = inputShapes;
        Shape classPredictionShape = new Shape();
        Shape anchorPredictionShape = new Shape();
        for (int i = 0; i < features.size(); i++) {
            childInputShapes = features.get(i).getOutputShapes(manager, childInputShapes);
            classPredictionShape =
                    concatShape(
                            classPredictionShape,
                            classPredictionShape(
                                    classPredictionBlocks.get(i)
                                            .getOutputShapes(manager, childInputShapes)[0]),
                            1);
            anchorPredictionShape =
                    concatShape(
                            anchorPredictionShape,
                            anchorPredictionShape(
                                    anchorPredictionBlocks.get(i)
                                            .getOutputShapes(manager, childInputShapes)[0]),
                            1);
        }
        return new Shape[] {classPredictionShape, anchorPredictionShape};
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

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
    }

    public static SequentialBlock getDownSamplingBlock(int numFilters) {
        SequentialBlock sequentialBlock = new SequentialBlock();
        for (int i = 0; i < 2; i++) {
            sequentialBlock
                    .add(
                            new Conv2D.Builder()
                                    .setKernel(new Shape(3, 3))
                                    .setNumFilters(numFilters)
                                    .optPad(new Shape(1, 1))
                                    .build())
                    .add(new BatchNorm.Builder().optEpsilon(2E-5f).build())
                    .add(Activation.reluBlock());
        }
        sequentialBlock.add(
                new LambdaBlock(
                        arrays ->
                                new NDList(
                                        Pool.maxPool(
                                                arrays.head(),
                                                new Shape(2, 2),
                                                new Shape(2, 2),
                                                new Shape(0, 0)))));
        return sequentialBlock;
    }

    public static Conv2D getClassPredictionBlock(int numAnchors, int numClasses) {
        return new Conv2D.Builder()
                .setKernel(new Shape(3, 3))
                .setNumFilters((numClasses + 1) * numAnchors)
                .optPad(new Shape(1, 1))
                .build();
    }

    public static Conv2D getAnchorPredictionBlock(int numAnchors) {
        return new Conv2D.Builder()
                .setKernel(new Shape(3, 3))
                .setNumFilters(4 * numAnchors)
                .optPad(new Shape(1, 1))
                .build();
    }

    public static class Builder {
        private NDManager manager;
        private Block network;
        private int numFeatures = -1;
        private List<Block> features;
        private List<List<Float>> sizes;
        private List<List<Float>> ratios;
        private List<Block> classPredictionBlocks = new ArrayList<>();
        private List<Block> anchorPredictionBlocks = new ArrayList<>();
        private int numClasses;
        private boolean useBatchNorm;
        private boolean globalPool = true;

        public Builder setManager(NDManager manager) {
            this.manager = manager;
            return this;
        }

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
        public Builder optNetwork(Block network) {
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
         * Sets the {@code Conv2D} blocks to be appended to the network to get multi-output network.
         *
         * @param features List of {@code Conv2D} blocks to be appended
         * @return Returns this Builder
         */
        public Builder optFeatures(List<Block> features) {
            this.features = features;
            return this;
        }

        /**
         * Sets the boolean whether to use BatchNorm layer after each attached convolutional layer.
         *
         * @param useBatchNorm Whether to use BatchNorm layer after each attached convolutional
         *     layer
         * @return Returns this Builder
         */
        public Builder optUseBatchNorm(boolean useBatchNorm) {
            this.useBatchNorm = useBatchNorm;
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

        public SingleShotDetection build() {
            if (manager == null) {
                throw new IllegalArgumentException("Manager must be set");
            }
            if (network == null) {
                SequentialBlock resnet =
                        (SequentialBlock)
                                new ResNetV1.Builder()
                                        .setImageShape(new Shape(1, 28, 28))
                                        .setNumLayers(50)
                                        .setOutSize(10)
                                        .build();
                resnet.removeLastBlock();
                resnet.removeLastBlock();
                resnet.removeLastBlock();
                resnet.removeLastBlock();
                network = resnet;
            }
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
                        new LambdaBlock(arrays -> new NDList(Pool.globalMaxPool(arrays.head()))));
            }
            int numberOfFeatureMaps = features.size();
            if (sizes.size() != ratios.size() && sizes.size() != numberOfFeatureMaps) {
                throw new IllegalArgumentException(
                        "Sizes and ratios must be of size: " + numberOfFeatureMaps);
            }
            for (int i = 0; i < numberOfFeatureMaps; i++) {
                int numAnchors = sizes.get(i).size() + ratios.get(i).size() - 1;
                classPredictionBlocks.add(getClassPredictionBlock(numAnchors, numClasses));
                anchorPredictionBlocks.add(getAnchorPredictionBlock(numAnchors));
            }
            return new SingleShotDetection(this);
        }
    }
}
