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
import ai.djl.nn.Activation;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;

/** Creates a block that performs all bert pretraining tasks (next sentence & masked language). */
public class BertPretrainingBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private final BertBlock bertBlock;
    private final BertMaskedLanguageModelBlock mlBlock;
    private final BertNextSentenceBlock nsBlock;

    /**
     * Creates a new Bert pretraining block fitting the given bert configuration.
     *
     * @param builder a builder with a bert configuration
     */
    public BertPretrainingBlock(final BertBlock.Builder builder) {
        super(VERSION);
        this.bertBlock = addChildBlock("Bert", builder.build());
        this.mlBlock =
                addChildBlock(
                        "BertMaskedLanguageModelBlock",
                        new BertMaskedLanguageModelBlock(bertBlock, Activation::gelu));
        this.nsBlock = addChildBlock("BertNextSentenceBlock", new BertNextSentenceBlock());
    }

    @Override
    public void initializeChildBlocks(
            final NDManager manager, final DataType dataType, final Shape... inputShapes) {
        inputNames = Arrays.asList("tokenIds", "typeIds", "sequenceMasks", "maskedIndices");
        Shape[] bertOutputShapes = bertBlock.initialize(manager, dataType, inputShapes);
        Shape embeddedSequence = bertOutputShapes[0];
        Shape pooledOutput = bertOutputShapes[1];
        Shape maskedIndices = inputShapes[2];
        Shape embeddingTableShape =
                new Shape(bertBlock.getTokenDictionarySize(), bertBlock.getEmbeddingSize());
        mlBlock.initialize(manager, dataType, embeddedSequence, embeddingTableShape, maskedIndices);
        nsBlock.initialize(manager, dataType, pooledOutput);
    }

    @Override
    public NDList forward(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        return forward(ps, inputs, training);
    }

    @Override
    public NDList forward(ParameterStore ps, NDList inputs, boolean training) {
        NDArray tokenIds = inputs.get(0);
        NDArray typeIds = inputs.get(1);
        NDArray sequenceMasks = inputs.get(2);
        NDArray maskedIndices = inputs.get(3);
        return forward(ps, tokenIds, typeIds, sequenceMasks, maskedIndices, training);
    }

    /**
     * Applies one bert pretraining step.
     *
     * @param ps the parameter store
     * @param tokenIds int, (B, S)
     * @param typeIds int, (B, S)
     * @param sequenceMasks int, (B, S)
     * @param maskedIndices int, (B, I)
     * @param training true=apply dropout etc.
     * @return next sentence probabilities (B, 2), masked token probabilities (B, I, D),
     */
    public NDList forward(
            final ParameterStore ps,
            final NDArray tokenIds,
            final NDArray typeIds,
            final NDArray sequenceMasks,
            final NDArray maskedIndices,
            final boolean training) {
        final MemoryScope scope =
                MemoryScope.from(tokenIds).add(typeIds, sequenceMasks, maskedIndices);
        // run the core bert model
        final NDList bertResult = bertBlock.forward(ps, tokenIds, typeIds, sequenceMasks, training);
        final NDArray embeddedSequence = bertResult.get(0);
        final NDArray pooledOutput = bertResult.get(1);
        // apply pooled output to the classifier
        final NDArray nextSentenceProbabilities = nsBlock.forward(ps, pooledOutput, training);
        // de-mask masked tokens
        final NDArray embeddingTable =
                bertBlock.getTokenEmbedding().getValue(ps, embeddedSequence.getDevice());
        final NDArray logProbs =
                mlBlock.forward(ps, embeddedSequence, maskedIndices, embeddingTable, training);

        scope.remove(tokenIds, typeIds, sequenceMasks, maskedIndices)
                .waitToRead(nextSentenceProbabilities, logProbs)
                .close();
        // return the next sentence & masked language result to apply the loss to
        return new NDList(nextSentenceProbabilities, logProbs);
    }

    /**
     * Returns the output shapes.
     *
     * @param inputShapes tokenIds int, (B, S), typeIds int, (B, S), sequenceMasks int, (B, S),
     *     maskedIndices int, (B, I)
     * @return next sentence probabilities (B, 2), masked token probabilities (B, I, D)
     */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        final long batchSize = inputShapes[0].get(0);
        final long maskedIndexCount = inputShapes[3].get(1);
        return new Shape[] {
            new Shape(batchSize, 2),
            new Shape(batchSize, maskedIndexCount, bertBlock.getTokenDictionarySize())
        };
    }
}
