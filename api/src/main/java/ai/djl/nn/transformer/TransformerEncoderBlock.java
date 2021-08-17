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
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Collections;
import java.util.function.Function;

/** Self-Attention based transformer encoder block. */
public class TransformerEncoderBlock extends AbstractBlock {

    /** The attention mechanism. */
    private ScaledDotProductAttentionBlock selfAttentionBlock;
    /** Dropout before residual & layer normalization. */
    private Dropout selfAttentionDropout;
    /** Normalization of attention output and residual. */
    private BatchNorm attentionNorm;
    /** Fully connected pointwise block for output projection. */
    private PointwiseFeedForwardBlock pointWisefullyConnected;
    /** Dropout after fully connected and before last residual & layer normalization. */
    private Dropout fullyConnectedDropout;
    /** Another normalization for the output and residual. */
    private BatchNorm outputNorm;

    /**
     * Creates a transformer encoder block.
     *
     * @param embeddingSize the embedding size for tokens
     * @param headCount number of attention blocks
     * @param hiddenSize the hidden size for fully connected networks
     * @param dropoutProbability dropout probability
     * @param activationFunction activation function
     */
    public TransformerEncoderBlock(
            int embeddingSize,
            int headCount,
            int hiddenSize,
            float dropoutProbability,
            Function<NDList, NDList> activationFunction) {
        this.selfAttentionBlock =
                addChildBlock(
                        "selfAttention",
                        ScaledDotProductAttentionBlock.builder()
                                .setEmbeddingSize(embeddingSize)
                                .setHeadCount(headCount)
                                .optAttentionProbsDropoutProb(dropoutProbability)
                                .build());
        this.selfAttentionDropout = Dropout.builder().optRate(dropoutProbability).build();
        this.attentionNorm = addChildBlock("attentionNorm", BatchNorm.builder().optAxis(2).build());
        this.pointWisefullyConnected =
                addChildBlock(
                        "outputBlock",
                        new PointwiseFeedForwardBlock(
                                Collections.singletonList(hiddenSize),
                                embeddingSize,
                                activationFunction));
        this.fullyConnectedDropout = Dropout.builder().optRate(dropoutProbability).build();
        this.outputNorm = addChildBlock("outputNorm", BatchNorm.builder().optAxis(2).build());
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        selfAttentionBlock.initialize(manager, dataType, inputShapes);
        attentionNorm.initialize(manager, dataType, inputShapes);
        pointWisefullyConnected.initialize(manager, dataType, inputShapes);
        outputNorm.initialize(manager, dataType, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray embedding = inputs.head();
        // perform attention lookup
        NDList attentionOutput = selfAttentionBlock.forward(ps, inputs, training);
        // add dropout to attention Output
        NDList attentionOutputAfterDropout =
                selfAttentionDropout.forward(ps, attentionOutput, training);
        // add input as residual
        NDArray withResidual = attentionOutputAfterDropout.singletonOrThrow().add(embedding);
        // apply normalization
        NDList normalized = attentionNorm.forward(ps, new NDList(withResidual), training);
        // apply pointwise projection
        NDList afterFullyConnected = pointWisefullyConnected.forward(ps, normalized, training);
        // apply dropout to fully connected output
        NDList afterFullyConnectedDropout =
                fullyConnectedDropout.forward(ps, afterFullyConnected, training);
        // add residual again
        NDList outputWithResidual =
                new NDList(afterFullyConnectedDropout.singletonOrThrow().add(embedding));
        // normalize result
        return outputNorm.forward(ps, new NDList(outputWithResidual), training);
    }
}
