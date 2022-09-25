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
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.nn.core.SparseMax;
import ai.djl.nn.norm.GhostBatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 */
public class TabNet {

    /**
     * Creates a FC->BN->GLU block used in tabNet.
     * In order to do GLU, we double the dimension of the input features to the GLU using a fc layer.
     * @param sharedBlock the shared fully connected layer
     * @param outDim the output feature dimension
     * @param virtualBatchSize the virtualBatchSize
     * @param batchNormMomentum the momentum used for ghost batchNorm layer
     * @return a FC->BN->GLU block
     */
    public static Block gluBlock(Block sharedBlock,int outDim,int virtualBatchSize,float batchNormMomentum){
        SequentialBlock featureBlock = new SequentialBlock();
        int units = 2*outDim;
        if(sharedBlock==null){
            featureBlock.add(Linear.builder().setUnits(units).build());
        }else{
            featureBlock.add(sharedBlock);
        }
        featureBlock.add(GhostBatchNorm.builder()
                .optVirtualBatchSize(virtualBatchSize)
                .optMomentum(batchNormMomentum)
                .build()).add(Activation.tabNetGLUBlock(outDim));

        return featureBlock;
    }


    /**
     * Creates a featureTransformer Block.
     * The feature transformer is where all the selected features are processed to generate the final output.
     * @param sharedBlocks the sharedBlocks of feature transformer
     * @param outDim the output dimension of feature transformer
     * @param numIndependent the number of independent blocks of feature transformer
     * @param virtualBatchSize the virtual batch size for ghost batch norm
     * @param batchNormMomentum the momentum for batch norm layer
     * @return a feature transformer
     */
    public static Block featureTransformer(List<Block> sharedBlocks,int outDim,int numIndependent,int virtualBatchSize,float batchNormMomentum){
        List<Block> allBlocks = new ArrayList<>();
        if(!sharedBlocks.isEmpty()){
            for(Block sharedBlock:sharedBlocks){
                allBlocks.add(gluBlock(sharedBlock,outDim,virtualBatchSize,batchNormMomentum));
            }
        }
        for(int i = 0;i<numIndependent;i++){
            allBlocks.add(gluBlock(null,outDim,virtualBatchSize,batchNormMomentum));
        }

        SequentialBlock featureBlocks = new SequentialBlock();
        int startIndex = 0;
        if(!sharedBlocks.isEmpty()){
            startIndex = 1;
            featureBlocks.add(allBlocks.get(0));
        }
        for(int i = startIndex;i<allBlocks.size();i++){
            featureBlocks.add(
                    new ParallelBlock(
                            ndLists -> {
                                NDList unit = ndLists.get(0);
                                NDList parallel = ndLists.get(1);
                                return new NDList(
                                        NDArrays.add(
                                                unit.singletonOrThrow(),
                                                parallel.singletonOrThrow()
                                        ).mul(Math.sqrt(0.5))
                                );
                            },Arrays.asList(allBlocks.get(i),Blocks.identityBlock())
                    )
            );
        }
        return featureBlocks;
    }

    /**
     * AttentionTransformer is where the tabNet models learn the relationship between relevant features,
     * and decides which features to pass on to the feature transformer of the current decision step.
     */
    public static class AttentionTransformer extends AbstractBlock{
        private static final Byte VERSION = 1;
        private Block fc;
        private Block bn;
        private Block sparseMax;

        /**
         * Creates an attentionTransformer Block with given parameters
         * @param units the units for fullyConnected Block
         * @param virtualBatchSize the virtual batch size for ghost batchNorm
         * @param batchNormMomentum the momentum for batchNorm layer
         */
        private AttentionTransformer(int units,int virtualBatchSize,float batchNormMomentum){
            super(VERSION);
            fc = addChildBlock("fullyConnected",Linear.builder().setUnits(units).build());
            bn = addChildBlock("ghostBatchNorm",GhostBatchNorm.builder().optVirtualBatchSize(virtualBatchSize).optMomentum(batchNormMomentum).build());
            sparseMax = addChildBlock("sparseMax",new SparseMax(-1,8));
        }

        @Override
        protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
            NDArray x = inputs.get(0), priors = inputs.get(1);
            NDList x1 = fc.forward(parameterStore,new NDList(x),training);
            NDList x2 = bn.forward(parameterStore,new NDList(x1),training);
            return sparseMax.forward(parameterStore,new NDList(x2.singletonOrThrow().mul(priors)),training);
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return new Shape[0];
        }

        @Override
        protected void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            Shape[] shapes = new Shape[]{inputShapes[0]};
            for (Block child : getChildren().values()) {
                child.initialize(manager, dataType, shapes);
                shapes = child.getOutputShapes(shapes);
            }
        }
    }

    public static class DecisionStep extends AbstractBlock{
        private static final Byte VERSION = 1;
        private Block featureTransformer;
        private Block attentionTransformer;

        public DecisionStep(int numD,int numA,List<Block> shared,int nInd,int virtualBatchSize,float batchNormMomentum){
            this.featureTransformer = addChildBlock("featureTransformer"
                    ,featureTransformer(shared,numD+numA,nInd,virtualBatchSize,batchNormMomentum));
            this.attentionTransformer = addChildBlock("attentionTransformer",
                    new AttentionTransformer(numA,virtualBatchSize,batchNormMomentum));
        }

        @Override
        protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
            NDArray x = inputs.get(0), a = inputs.get(1), priors = inputs.get(2);
            NDList mask = attentionTransformer.forward(parameterStore,new NDList(a,priors),training);
            NDArray sparseLoss = mask.singletonOrThrow().mul(-1).mul(NDArrays.add(mask.singletonOrThrow(),1e-10).log()).mean();
            NDList x1 = featureTransformer.forward(parameterStore,new NDList(x),training);
            return new NDList(x1.singletonOrThrow(),sparseLoss);
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return new Shape[0];
        }

        @Override
        protected void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            Shape[] shapes = new Shape[]{inputShapes[0]};
            for (Block child : getChildren().values()) {
                child.initialize(manager, dataType, shapes);
                shapes = child.getOutputShapes(shapes);
            }
        }
    }


    /** The Builder to construct a {@link TabNet} object. */
    public static class Builder{
        int inputDim;
        int finalOutDim;
        int virtualBatchSize = 128;
        float batchNormMomentum = 0.9f;

        /**
         * Sets the virtual batch size for ghost batch norm
         * @param virtualBatchSize the virtual batch size
         * @return this {@code Builder}
         */
        public Builder optVirtualBatchSize(int virtualBatchSize){
            this.virtualBatchSize = virtualBatchSize;
            return this;
        }

        /**
         * Builds an attentionTransformer Block for testing with problem
         * @return an attentionTransformer Block
         */
        public Block buildAttentionTransformer(){
            return new AttentionTransformer(10,virtualBatchSize,batchNormMomentum);
        }
    }
}
