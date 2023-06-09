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
package ai.djl.basicmodelzoo.tabular;

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
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.core.SparseMax;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.GhostBatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * {@code TabNet} contains a generic implementation of TabNet adapted from
 * https://towardsdatascience.com/implementing-tabnet-in-pytorch-fc977c383279 (Original author
 * Samrat Thapa)
 *
 * <p>TabNet is a neural architecture for tabular dataset developed by the research team at Google
 * Cloud AI. It was able to achieve state_of_the_art results on several datasets in both regression
 * and classification problems. Another desirable feature of TabNet is interpretability. Contrary to
 * most of deep learning, where the neural networks act like black boxes, we can interpret which
 * features the models selects in case of TabNet.
 *
 * <p>see https://arxiv.org/pdf/1908.07442.pdf for more information about TabNet
 */
public final class TabNet extends AbstractBlock {
    private static final byte VERSION = 1;
    private Block firstStep;
    private List<Block> steps;
    private Block fullyConnected;
    private Block batchNorm;
    private int numD;
    private int numA;

    /**
     * Creates a {@link TabNet} instance with given builder.
     *
     * @param builder the builder to create TabNet
     */
    private TabNet(Builder builder) {
        super(VERSION);
        batchNorm =
                addChildBlock(
                        "batchNorm",
                        BatchNorm.builder().optMomentum(builder.batchNormMomentum).build());
        List<Block> sharedBlocks = new ArrayList<>();
        for (int i = 0; i < builder.numShared; i++) {
            sharedBlocks.add(
                    addChildBlock(
                            "sharedfc" + i,
                            Linear.builder().setUnits(2L * (builder.numA + builder.numD)).build()));
        }

        firstStep =
                addChildBlock(
                        "featureTransformer",
                        featureTransformer(
                                sharedBlocks,
                                builder.numD + builder.numA,
                                builder.numIndependent,
                                builder.virtualBatchSize,
                                builder.batchNormMomentum));

        steps = new ArrayList<>();
        for (int i = 0; i < builder.numSteps - 1; i++) {
            steps.add(
                    addChildBlock(
                            "steps" + (i + 1),
                            new DecisionStep(
                                    builder.inputDim,
                                    builder.numD,
                                    builder.numA,
                                    sharedBlocks,
                                    builder.numIndependent,
                                    builder.virtualBatchSize,
                                    builder.batchNormMomentum)));
        }
        fullyConnected =
                addChildBlock(
                        "fullyConnected", Linear.builder().setUnits(builder.finalOutDim).build());
        this.numD = builder.numD;
        this.numA = builder.numA;
    }

    /**
     * Applies tabNetGLU activation(which is mostly used in tabNet) on the input {@link NDArray}.
     *
     * @param array the input {@link NDArray}
     * @param units the half number of the resultant features
     * @return the {@link NDArray} after applying tabNetGLU function
     */
    public static NDArray tabNetGLU(NDArray array, int units) {
        return array.get(":,:{}", units).mul(Activation.sigmoid(array.get(":, {}:", units)));
    }

    /**
     * Applies tabNetGLU activation(which is mostly used in tabNet) on the input singleton {@link
     * NDList}.
     *
     * @param arrays the input singleton {@link NDList}
     * @param units the half number of the resultant features
     * @return the singleton {@link NDList} after applying tabNetGLU function
     */
    public static NDList tabNetGLU(NDList arrays, int units) {
        return new NDList(tabNetGLU(arrays.singletonOrThrow(), units));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #tabNetGLU(NDArray, int)} activation
     * function in its forward function.
     *
     * @param units the half number of feature
     * @return {@link LambdaBlock} that applies the {@link #tabNetGLU(NDArray, int)} activation
     *     function
     */
    public static Block tabNetGLUBlock(int units) {
        return new LambdaBlock(arrays -> tabNetGLU(arrays, units), "tabNetGLU");
    }

    /**
     * Creates a FC-BN-GLU block used in tabNet. In order to do GLU, we double the dimension of the
     * input features to the GLU using a fc layer.
     *
     * @param sharedBlock the shared fully connected layer
     * @param outDim the output feature dimension
     * @param virtualBatchSize the virtualBatchSize
     * @param batchNormMomentum the momentum used for ghost batchNorm layer
     * @return a FC-BN-GLU block
     */
    public static Block gluBlock(
            Block sharedBlock, int outDim, int virtualBatchSize, float batchNormMomentum) {
        SequentialBlock featureBlock = new SequentialBlock();
        int units = 2 * outDim;
        if (sharedBlock == null) {
            featureBlock.add(Linear.builder().setUnits(units).build());
        } else {
            featureBlock.add(sharedBlock);
        }
        featureBlock
                .add(
                        GhostBatchNorm.builder()
                                .optVirtualBatchSize(virtualBatchSize)
                                .optMomentum(batchNormMomentum)
                                .build())
                .add(tabNetGLUBlock(outDim));

        return featureBlock;
    }

    /**
     * Creates a featureTransformer Block. The feature transformer is where all the selected
     * features are processed to generate the final output.
     *
     * @param sharedBlocks the sharedBlocks of feature transformer
     * @param outDim the output dimension of feature transformer
     * @param numIndependent the number of independent blocks of feature transformer
     * @param virtualBatchSize the virtual batch size for ghost batch norm
     * @param batchNormMomentum the momentum for batch norm layer
     * @return a feature transformer
     */
    public static Block featureTransformer(
            List<Block> sharedBlocks,
            int outDim,
            int numIndependent,
            int virtualBatchSize,
            float batchNormMomentum) {
        List<Block> allBlocks = new ArrayList<>();
        if (!sharedBlocks.isEmpty()) {
            for (Block sharedBlock : sharedBlocks) {
                allBlocks.add(gluBlock(sharedBlock, outDim, virtualBatchSize, batchNormMomentum));
            }
        }
        for (int i = 0; i < numIndependent; i++) {
            allBlocks.add(gluBlock(null, outDim, virtualBatchSize, batchNormMomentum));
        }

        SequentialBlock featureBlocks = new SequentialBlock();
        int startIndex = 0;
        if (!sharedBlocks.isEmpty()) {
            startIndex = 1;
            featureBlocks.add(allBlocks.get(0));
        }
        for (int i = startIndex; i < allBlocks.size(); i++) {
            featureBlocks.add(
                    new ParallelBlock(
                            ndLists -> {
                                NDList unit = ndLists.get(0);
                                NDList parallel = ndLists.get(1);
                                return new NDList(
                                        NDArrays.add(
                                                        unit.singletonOrThrow(),
                                                        parallel.singletonOrThrow())
                                                .mul(Math.sqrt(0.5)));
                            },
                            Arrays.asList(allBlocks.get(i), Blocks.identityBlock())));
        }
        return featureBlocks;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDManager manager = inputs.getManager();
        NDArray input = inputs.singletonOrThrow();
        input = input.reshape(input.size(0), input.size() / input.size(0)); // batch flatten
        NDArray x =
                batchNorm.forward(parameterStore, new NDList(input), training).singletonOrThrow();
        NDArray xa =
                firstStep
                        .forward(parameterStore, new NDList(x), training)
                        .singletonOrThrow()
                        .get(":," + this.numD + ":");
        NDArray sparseLoss = null;
        NDArray out = null;
        NDArray priors = manager.ones(x.getShape());
        for (Block step : steps) {
            NDList tempRes = step.forward(parameterStore, new NDList(x, xa, priors), training);
            NDArray xte = tempRes.get(0);
            NDArray loss = tempRes.get(1);
            if (out == null) {
                out = Activation.relu(xte.get(":,:" + this.numD));
            } else {
                out = out.add(Activation.relu(xte.get(":,:" + this.numD)));
            }
            xa = xte.get(":," + this.numD + ":");
            sparseLoss = sparseLoss == null ? loss : sparseLoss.add(loss);
        }
        NDArray finalOutput =
                fullyConnected
                        .forward(parameterStore, new NDList(out), training)
                        .singletonOrThrow();
        return new NDList(finalOutput, sparseLoss);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape[] shapes = inputShapes;
        Shape[] xShapes = batchNorm.getOutputShapes(shapes);

        Shape[] xaShapes = firstStep.getOutputShapes(xShapes); // input shape for xa
        xaShapes[0] = Shape.update(xaShapes[0], xaShapes[0].dimension() - 1, this.numA);
        shapes =
                new Shape[] {
                    xShapes[0], xaShapes[0], xShapes[0]
                }; // shape of priors should be the same as x
        Shape outputShape = new Shape();
        Shape lossShape = new Shape();
        for (Block step : steps) {
            Shape[] outputShapes = step.getOutputShapes(shapes);
            outputShape = Shape.update(outputShapes[0], outputShapes[0].dimension() - 1, numD);
            lossShape = outputShapes[1];
        }
        outputShape = fullyConnected.getOutputShapes(new Shape[] {outputShape})[0];
        return new Shape[] {outputShape, lossShape};
    }

    /** {@inheritDoc} */
    @Override
    protected void initializeChildBlocks(
            NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] shapes = inputShapes;
        batchNorm.initialize(manager, dataType, shapes);
        Shape[] xShapes = batchNorm.getOutputShapes(shapes);
        firstStep.initialize(manager, dataType, xShapes);
        Shape[] xaShapes = firstStep.getOutputShapes(xShapes); // input shape for xa
        xaShapes[0] = Shape.update(xaShapes[0], xaShapes[0].dimension() - 1, this.numD);
        shapes =
                new Shape[] {
                    xShapes[0], xaShapes[0], xShapes[0]
                }; // shape of priors should be the same as x
        Shape outputShape = new Shape();
        for (Block step : steps) {
            step.initialize(manager, dataType, shapes);
            Shape[] outputShapes = step.getOutputShapes(shapes);
            outputShape = Shape.update(outputShapes[0], outputShapes[0].dimension() - 1, numD);
        }
        fullyConnected.initialize(manager, dataType, outputShape);
    }

    /**
     * Creates a builder to build a {@link TabNet}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * AttentionTransformer is where the tabNet models learn the relationship between relevant
     * features, and decides which features to pass on to the feature transformer of the current
     * decision step.
     */
    public static final class AttentionTransformer extends AbstractBlock {
        private static final Byte VERSION = 1;
        private Block fullyConnected;
        private Block batchNorm;
        private Block sparseMax;

        /**
         * Creates an attentionTransformer Block with given parameters.
         *
         * @param inputDim the input Dimension of the TabNet
         * @param virtualBatchSize the virtual batch size for ghost batchNorm
         * @param batchNormMomentum the momentum for batchNorm layer
         */
        private AttentionTransformer(int inputDim, int virtualBatchSize, float batchNormMomentum) {
            super(VERSION);
            fullyConnected =
                    addChildBlock("fullyConnected", Linear.builder().setUnits(inputDim).build());
            batchNorm =
                    addChildBlock(
                            "ghostBatchNorm",
                            GhostBatchNorm.builder()
                                    .optVirtualBatchSize(virtualBatchSize)
                                    .optMomentum(batchNormMomentum)
                                    .build());
            sparseMax = addChildBlock("sparseMax", new SparseMax());
        }

        /** {@inheritDoc} */
        @Override
        protected NDList forwardInternal(
                ParameterStore parameterStore,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            NDArray x = inputs.get(0);
            NDArray priors = inputs.get(1);
            NDList x1 = fullyConnected.forward(parameterStore, new NDList(x), training);
            NDList x2 = batchNorm.forward(parameterStore, x1, training);
            return sparseMax.forward(
                    parameterStore, new NDList(x2.singletonOrThrow().mul(priors)), training);
        }

        /** {@inheritDoc} */
        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            Shape[] shapes = {inputShapes[0]};
            for (Pair<String, Block> child : getChildren()) {
                shapes = child.getValue().getOutputShapes(shapes);
            }
            return shapes;
        }

        /** {@inheritDoc} */
        @Override
        protected void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            Shape[] shapes = {inputShapes[0]};
            for (Block child : getChildren().values()) {
                child.initialize(manager, dataType, shapes);
                shapes = child.getOutputShapes(shapes);
            }
        }
    }

    /** DecisionStep is just combining featureTransformer and attentionTransformer together. */
    public static final class DecisionStep extends AbstractBlock {
        private static final Byte VERSION = 1;
        private Block featureTransformer;
        private Block attentionTransformer;

        /**
         * Creates a {@link DecisionStep} with given parameters.
         *
         * @param inputDim the number of input dimension for attentionTransformer
         * @param numD the number of dimension except attentionTransformer
         * @param numA the number of dimension for attentionTransformer
         * @param shared the shared fullyConnected layers
         * @param nInd the number of independent fullyConnected layers
         * @param virtualBatchSize the virtual batch size
         * @param batchNormMomentum the momentum for batchNorm layer
         */
        public DecisionStep(
                int inputDim,
                int numD,
                int numA,
                List<Block> shared,
                int nInd,
                int virtualBatchSize,
                float batchNormMomentum) {
            super(VERSION);
            this.featureTransformer =
                    addChildBlock(
                            "featureTransformer",
                            featureTransformer(
                                    shared,
                                    numD + numA,
                                    nInd,
                                    virtualBatchSize,
                                    batchNormMomentum));
            this.attentionTransformer =
                    addChildBlock(
                            "attentionTransformer",
                            new AttentionTransformer(
                                    inputDim, virtualBatchSize, batchNormMomentum));
        }

        /** {@inheritDoc} */
        @Override
        protected NDList forwardInternal(
                ParameterStore parameterStore,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            NDArray x = inputs.get(0);
            NDArray a = inputs.get(1);
            NDArray priors = inputs.get(2);
            NDList mask =
                    attentionTransformer.forward(parameterStore, new NDList(a, priors), training);
            NDArray sparseLoss =
                    mask.singletonOrThrow()
                            .mul(-1)
                            .mul(NDArrays.add(mask.singletonOrThrow(), 1e-10).log());
            NDList x1 = featureTransformer.forward(parameterStore, new NDList(x), training);
            return new NDList(x1.singletonOrThrow(), sparseLoss);
        }

        /** {@inheritDoc} */
        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            Shape[] xShape = {inputShapes[0]};
            Shape[] aShape = {inputShapes[1], inputShapes[2]};
            Shape[] x1Shape = featureTransformer.getOutputShapes(xShape);
            Shape[] lossShape = attentionTransformer.getOutputShapes(aShape);
            return new Shape[] {x1Shape[0], lossShape[0]};
        }

        /** {@inheritDoc} */
        @Override
        protected void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            Shape[] xShape = {inputShapes[0]};
            Shape[] aShape = {inputShapes[1], inputShapes[2]};
            this.attentionTransformer.initialize(manager, dataType, aShape);
            this.featureTransformer.initialize(manager, dataType, xShape);
        }
    }

    /** The Builder to construct a {@link TabNet} object. */
    public static class Builder {
        int inputDim = 128;
        int finalOutDim = 10;
        int numD = 64;
        int numA = 64;
        int numShared = 2;
        int numIndependent = 2;
        int numSteps = 5;
        int virtualBatchSize = 128;
        float batchNormMomentum = 0.9f;

        /**
         * Sets the input dimension of TabNet.
         *
         * @param inputDim the input dimension
         * @return this {@code Builder}
         */
        public Builder setInputDim(int inputDim) {
            this.inputDim = inputDim;
            return this;
        }

        /**
         * Sets the output dimension for TabNet.
         *
         * @param outDim the output dimension
         * @return this {@code Builder}
         */
        public Builder setOutDim(int outDim) {
            this.finalOutDim = outDim;
            return this;
        }

        /**
         * Sets the number of dimension except attentionTransformer.
         *
         * @param numD the number of dimension except attentionTransformer
         * @return this {@code Builder}
         */
        public Builder optNumD(int numD) {
            this.numD = numD;
            return this;
        }

        /**
         * Sets the number of dimension for attentionTransformer.
         *
         * @param numA the number of dimension for attentionTransformer
         * @return this {@code Builder}
         */
        public Builder optNumA(int numA) {
            this.numA = numA;
            return this;
        }

        /**
         * Sets the number of shared fullyConnected layers.
         *
         * @param numShared the number of shared fullyConnected layers
         * @return this {@code Builder}
         */
        public Builder optNumShared(int numShared) {
            this.numShared = numShared;
            return this;
        }

        /**
         * Sets the number of independent fullyConnected layers.
         *
         * @param numIndependent the number of independent fullyConnected layers
         * @return this {@code Builder}
         */
        public Builder optNumIndependent(int numIndependent) {
            this.numIndependent = numIndependent;
            return this;
        }

        /**
         * Sets the number of decision steps for tabNet.
         *
         * @param numSteps the number of decision steps for tabNet
         * @return this {@code Builder}
         */
        public Builder optNumSteps(int numSteps) {
            this.numSteps = numSteps;
            return this;
        }

        /**
         * Sets the virtual batch size for ghost batch norm.
         *
         * @param virtualBatchSize the virtual batch size
         * @return this {@code Builder}
         */
        public Builder optVirtualBatchSize(int virtualBatchSize) {
            this.virtualBatchSize = virtualBatchSize;
            return this;
        }

        /**
         * Sets the momentum for batchNorm layer.
         *
         * @param batchNormMomentum the momentum for batchNormLayer
         * @return this {@code Builder}
         */
        public Builder optBatchNormMomentum(float batchNormMomentum) {
            this.batchNormMomentum = batchNormMomentum;
            return this;
        }

        /**
         * Builds an attentionTransformer with given parameter for test.
         *
         * @param units the number of test units
         * @return an attentionTransformer Block
         */
        public Block buildAttentionTransformer(int units) {
            return new AttentionTransformer(10, virtualBatchSize, batchNormMomentum);
        }

        /**
         * Builds a TabNet with given {@code Builder}.
         *
         * @return a tabNetBlock
         */
        public Block build() {
            return new TabNet(this);
        }
    }
}
