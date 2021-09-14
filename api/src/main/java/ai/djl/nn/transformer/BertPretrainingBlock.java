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

/** Creates a block that performs all bert pretraining tasks (next sentence and masked language). */
public class BertPretrainingBlock extends AbstractBlock {

    private BertBlock bertBlock;
    private BertMaskedLanguageModelBlock mlBlock;
    private BertNextSentenceBlock nsBlock;

    /**
     * Creates a new Bert pretraining block fitting the given bert configuration.
     *
     * @param builder a builder with a bert configuration
     */
    public BertPretrainingBlock(final BertBlock.Builder builder) {
        this.bertBlock = addChildBlock("Bert", builder.build());
        this.mlBlock =
                addChildBlock(
                        "BertMaskedLanguageModelBlock",
                        new BertMaskedLanguageModelBlock(bertBlock, Activation::gelu));
        this.nsBlock = addChildBlock("BertNextSentenceBlock", new BertNextSentenceBlock());
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        inputNames = Arrays.asList("tokenIds", "typeIds", "sequenceMasks", "maskedIndices");
        bertBlock.initialize(manager, dataType, inputShapes);
        Shape[] bertOutputShapes = bertBlock.getOutputShapes(inputShapes);
        Shape embeddedSequence = bertOutputShapes[0];
        Shape pooledOutput = bertOutputShapes[1];
        Shape maskedIndices = inputShapes[2];
        Shape embeddingTableShape =
                new Shape(bertBlock.getTokenDictionarySize(), bertBlock.getEmbeddingSize());
        mlBlock.initialize(manager, dataType, embeddedSequence, embeddingTableShape, maskedIndices);
        nsBlock.initialize(manager, dataType, pooledOutput);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray tokenIds = inputs.get(0);
        NDArray typeIds = inputs.get(1);
        NDArray sequenceMasks = inputs.get(2);
        NDArray maskedIndices = inputs.get(3);
        try (NDManager scope = NDManager.subManagerOf(tokenIds)) {
            scope.tempAttachAll(inputs);
            // run the core bert model
            NDList bertResult =
                    bertBlock.forward(ps, new NDList(tokenIds, typeIds, sequenceMasks), training);
            NDArray embeddedSequence = bertResult.get(0);
            NDArray pooledOutput = bertResult.get(1);
            // apply pooled output to the classifier
            NDArray nextSentenceProbabilities =
                    nsBlock.forward(ps, new NDList(pooledOutput), training).singletonOrThrow();
            // de-mask masked tokens
            NDArray embeddingTable =
                    bertBlock
                            .getTokenEmbedding()
                            .getValue(ps, embeddedSequence.getDevice(), training);
            NDArray logProbs =
                    mlBlock.forward(
                                    ps,
                                    new NDList(embeddedSequence, maskedIndices, embeddingTable),
                                    training)
                            .singletonOrThrow();

            // return the next sentence & masked language result to apply the loss to
            return scope.ret(new NDList(nextSentenceProbabilities, logProbs));
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        long batchSize = inputShapes[0].get(0);
        long maskedIndexCount = inputShapes[3].get(1);
        return new Shape[] {
            new Shape(batchSize, 2),
            new Shape(batchSize, maskedIndexCount, bertBlock.getTokenDictionarySize())
        };
    }
}
