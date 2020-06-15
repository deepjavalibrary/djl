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
import ai.djl.nn.ParameterType;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;
import java.util.function.Function;

/** Block for the bert masked language task. */
public class BertMaskedLanguageModelBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private final Linear sequenceProjection;

    private final BatchNorm sequenceNorm;

    private final Parameter dictionaryBias;

    private final Function<NDArray, NDArray> hiddenActivation;

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
                                .setOutChannels(bertBlock.getEmbeddingSize())
                                .optBias(true)
                                .build());
        this.sequenceNorm = addChildBlock("sequenceNorm", BatchNorm.builder().optAxis(1).build());
        this.dictionaryBias =
                addParameter(
                        new Parameter("dictionaryBias", this, ParameterType.BIAS),
                        new Shape(bertBlock.getTokenDictionarySize()));
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
                indices.add(sequenceOffsets).reshape(1, batchSize * indicesPerSequence);
        // Now we create one long sequence by appending all sequences
        NDArray flattenedSequences = sequences.reshape(batchSize * sequenceLength, width);
        // We use the absolute indices to gather the elements of the flattened sequences
        return MissingOps.gatherNd(flattenedSequences, absoluteIndices);
    }

    @Override
    public void initializeChildBlocks(
            final NDManager manager, final DataType dataType, final Shape... inputShapes) {
        inputNames = Arrays.asList("sequence", "maskedIndices", "embeddingTable");
        final int width = (int) inputShapes[0].get(2);
        sequenceProjection.initialize(manager, dataType, new Shape(-1, width));
        sequenceNorm.initialize(manager, dataType, new Shape(-1, width));
    }

    @Override
    public NDList forward(
            final ParameterStore ps,
            final NDList inputs,
            final boolean training,
            final PairList<String, Object> params) {
        return forward(ps, inputs, training);
    }

    @Override
    public NDList forward(final ParameterStore ps, final NDList inputs, final boolean training) {
        final NDArray sequenceOutput = inputs.get(0); // (B, S, E)
        final NDArray maskedIndices = inputs.get(1); // (B, I)
        final NDArray embeddingTable = inputs.get(2); // (D, E)
        final NDArray logProbs =
                forward(ps, sequenceOutput, maskedIndices, embeddingTable, training);
        return new NDList(logProbs);
    }

    /**
     * Calculates the result of the masked language task.
     *
     * @param ps the parameter store
     * @param sequenceOutput The sequence output of the bert model (B, S, E)
     * @param maskedIndices The indices of the tokens masked for pretraining (B, I)
     * @param embeddingTable The embedding table of the bert model (D, E)
     * @param training true=apply dropout etc.
     * @return the log probabilities for each dictionary item for each masked token, size (B * I, D)
     */
    public NDArray forward(
            final ParameterStore ps,
            final NDArray sequenceOutput,
            final NDArray maskedIndices,
            final NDArray embeddingTable,
            final boolean training) {
        MemoryScope scope = MemoryScope.from(sequenceOutput).add(maskedIndices);
        final NDArray gatheredTokens =
                gatherFromIndices(sequenceOutput, maskedIndices); // (B * I, E)
        final NDArray projectedTokens =
                hiddenActivation.apply(
                        sequenceProjection
                                .forward(ps, new NDList(gatheredTokens), training)
                                .head()); // (B * I, E)
        final NDArray normalizedTokens =
                sequenceNorm
                        .forward(ps, new NDList(projectedTokens), training)
                        .head(); // (B * I, E)
        // raw logits for each position to correspond to an entry in the embedding table
        final NDArray embeddingTransposed = embeddingTable.transpose();
        embeddingTransposed.attach(gatheredTokens.getManager());
        final NDArray logits = normalizedTokens.dot(embeddingTransposed); // (B * I, D)
        // we add an offset for each dictionary entry
        final NDArray logitsWithBias =
                logits.add(ps.getValue(dictionaryBias, logits.getDevice())); // (B * I, D)
        // now we apply log Softmax to get proper log probabilities
        final NDArray logProbs = logitsWithBias.logSoftmax(1); // (B * I, D)

        scope.remove(sequenceOutput, maskedIndices).waitToRead(logProbs).close();

        return logProbs;
    }

    @Override
    public Shape[] getOutputShapes(final NDManager manager, final Shape[] inputShapes) {
        final int batchSize = (int) inputShapes[0].get(0);
        final int indexCount = (int) inputShapes[1].get(1);
        final int dictionarySize = (int) inputShapes[2].get(0);
        return new Shape[] {new Shape(batchSize * indexCount, dictionarySize)};
    }
}
