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
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;
import java.util.function.Function;

/** Block for the bert masked language task. */
public class BertMaskedLanguageModelBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private Linear sequenceProjection;

    private BatchNorm sequenceNorm;

    private Parameter dictionaryBias;

    private Function<NDArray, NDArray> hiddenActivation;

    /**
     * Creates a new block that applies the masked language task.
     *
     * @param bertBlock the bert block to create the task for
     * @param hiddenActivation the activation to use for the hidden layer
     */
    public BertMaskedLanguageModelBlock(
            BertBlock bertBlock, Function<NDArray, NDArray> hiddenActivation) {
        super(VERSION);
        this.sequenceProjection =
                addChildBlock(
                        "sequenceProjection",
                        Linear.builder()
                                .setUnits(bertBlock.getEmbeddingSize())
                                .optBias(true)
                                .build());
        this.sequenceNorm = addChildBlock("sequenceNorm", BatchNorm.builder().optAxis(1).build());
        this.dictionaryBias =
                addParameter(
                        Parameter.builder()
                                .setName("dictionaryBias")
                                .setType(Parameter.Type.BIAS)
                                .optShape(new Shape(bertBlock.getTokenDictionarySize()))
                                .build());
        this.hiddenActivation = hiddenActivation;
    }

    /**
     * Given a 3D array of shape (B, S, E) and a 2D array of shape (B, I) returns the flattened
     * lookup result of shape (B * I * E).
     *
     * @param sequences Sequences of embeddings
     * @param indices Indices into the sequences. The indices are relative within each sequence,
     *     i.e. [[0, 1],[0, 1]] would return the first two elements of two sequences.
     * @return The flattened result of gathering elements from the sequences
     */
    public static NDArray gatherFromIndices(NDArray sequences, NDArray indices) {
        int batchSize = (int) sequences.getShape().get(0);
        int sequenceLength = (int) sequences.getShape().get(1);
        int width = (int) sequences.getShape().get(2);
        int indicesPerSequence = (int) indices.getShape().get(1);
        // this creates a list of offsets for each sequence. Say sequence length is 16 and
        // batch size is 4, this creates [0, 16, 32, 48]. Each
        NDArray sequenceOffsets =
                indices.getManager()
                        .newSubManager(indices.getDevice())
                        .arange(0, batchSize) // [0, 1, 2, ..., batchSize - 1]
                        .mul(sequenceLength) // [0, 16, 32, ...]
                        .reshape(batchSize, 1); // [[0], [16], [32], ...]
        // The following adds the sequence offsets to every index for every sequence.
        // This works, because the single values in the sequence offsets are propagated
        NDArray absoluteIndices =
                indices.add(sequenceOffsets).reshape(1, (long) batchSize * indicesPerSequence);
        // Now we create one long sequence by appending all sequences
        NDArray flattenedSequences = sequences.reshape((long) batchSize * sequenceLength, width);
        // We use the absolute indices to gather the elements of the flattened sequences
        return MissingOps.gatherNd(flattenedSequences, absoluteIndices);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        inputNames = Arrays.asList("sequence", "maskedIndices", "embeddingTable");
        int width = (int) inputShapes[0].get(2);
        sequenceProjection.initialize(manager, dataType, new Shape(-1, width));
        sequenceNorm.initialize(manager, dataType, new Shape(-1, width));
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray sequenceOutput = inputs.get(0); // (B, S, E)
        NDArray maskedIndices = inputs.get(1); // (B, I)
        NDArray embeddingTable = inputs.get(2); // (D, E)
        try (NDManager scope = NDManager.subManagerOf(sequenceOutput)) {
            scope.tempAttachAll(sequenceOutput, maskedIndices);
            NDArray gatheredTokens = gatherFromIndices(sequenceOutput, maskedIndices); // (B * I, E)
            NDArray projectedTokens =
                    hiddenActivation.apply(
                            sequenceProjection
                                    .forward(ps, new NDList(gatheredTokens), training)
                                    .head()); // (B * I, E)
            NDArray normalizedTokens =
                    sequenceNorm
                            .forward(ps, new NDList(projectedTokens), training)
                            .head(); // (B * I, E)
            // raw logits for each position to correspond to an entry in the embedding table
            NDArray embeddingTransposed = embeddingTable.transpose();
            embeddingTransposed.attach(gatheredTokens.getManager());
            NDArray logits = normalizedTokens.dot(embeddingTransposed); // (B * I, D)
            // we add an offset for each dictionary entry
            NDArray logitsWithBias =
                    logits.add(
                            ps.getValue(
                                    dictionaryBias, logits.getDevice(), training)); // (B * I, D)
            // now we apply log Softmax to get proper log probabilities
            NDArray logProbs = logitsWithBias.logSoftmax(1); // (B * I, D)

            return scope.ret(new NDList(logProbs));
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(final Shape[] inputShapes) {
        int batchSize = (int) inputShapes[0].get(0);
        int indexCount = (int) inputShapes[1].get(1);
        int dictionarySize = (int) inputShapes[2].get(0);
        return new Shape[] {new Shape((long) batchSize * indexCount, dictionarySize)};
    }
}
