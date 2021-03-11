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
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * A Block implementing scaled product attention according to <a
 * href="https://arxiv.org/abs/1706.03762">Vaswani et. al.</a>.
 *
 * <p>Abbreviations used:
 *
 * <ul>
 *   <li>E = embedding size
 *   <li>B = batch size
 *   <li>N = number of attention heads
 *   <li>F = "from" sequence length (key/value sequence), the input sequence
 *   <li>T = "to" sequence length (query sequence), e.g. the length of the output sequence
 *   <li>S = a sequence length, either F or T
 *   <li>H = Attention head size (= E / N)
 * </ul>
 *
 * <p>In many use cases F=T. For self attention, the input is equal to the output.
 *
 * <p>This block can process input in four forms:
 *
 * <ul>
 *   <li>Input size one: [Values] = [(B, F, E)], only input is used as key, query and value
 *       (unmasked self attention), e.g. BERT
 *   <li>Input size two: [Values, Mask] = [(B, F, E), (B, F, F)], first input is used as key, query
 *       and value, masked self attention
 *   <li>Input size three: [Keys, Queries, Values] = [(B, F, E), (B, T, E), (B, F, E)], inputs are
 *       interpreted as keys, queries and values, unmasked attention
 *   <li>Input size four: [Keys, Queries, Values, Mask] = [(B, F, E), (B, T, E), (B, F, E), (B, T,
 *       F)], inputs are interpreted as keys, queries, values and an attention mask, full masked
 *       attention.
 * </ul>
 *
 * <p>Attention masks must contain a 1 for positions to keep and a 0 for positions to mask.
 */
// We name local variables for tensor dimensions as in the paper and the reference code.
// While against the general code style, it makes things much easier readable here.
@SuppressWarnings({
    "LocalVariableName",
    "PMD.LocalVariableNamingConventions",
    "ParameterName",
    "PMD.FormalParameterNamingConventions"
})
public final class ScaledDotProductAttentionBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    /** Size of the Word-/Token-embeddings we use the attention on. */
    private int embeddingSize;
    /** Number of attention heads. */
    private int headCount;
    /** Pointwise Linear projection of the keys. */
    private Linear keyProjection;
    /** Pointwise Linear projection of the queries. */
    private Linear queryProjection;
    /** Pointwise Linear projection of the values. */
    private Linear valueProjection;
    /** Pointwise Linear projection of the results. */
    private Linear resultProjection;
    /** Dropout operation to be applied after probability calculation. */
    private Dropout attentionProbsDropout;

    private ScaledDotProductAttentionBlock(Builder builder) {
        super(VERSION);

        this.embeddingSize = builder.embeddingSize;
        this.headCount = builder.headCount;

        this.keyProjection = addChildBlock("keyProjection", buildProjection());
        this.queryProjection = addChildBlock("queryProjection", buildProjection());
        this.valueProjection = addChildBlock("valueProjection", buildProjection());
        this.resultProjection = addChildBlock("resultProjection", buildProjection());

        this.attentionProbsDropout =
                addChildBlock(
                        "probabilityDropout",
                        Dropout.builder().optRate(builder.attentionProbsDropoutProb).build());
    }

    /**
     * Helper method to build a pointwise linear projection for the current embedding size.
     *
     * @return a linear projection with bias and an output size equal to the embedding size.
     */
    private Linear buildProjection() {
        return Linear.builder().setUnits(embeddingSize).optBias(true).build();
    }

    /**
     * Pointwise Linear projection of the keys.
     *
     * @return Pointwise Linear projection of the keys.
     */
    public Linear getKeyProjection() {
        return keyProjection;
    }
    /**
     * Pointwise Linear projection of the queries.
     *
     * @return Pointwise Linear projection of the queries.
     */
    public Linear getQueryProjection() {
        return queryProjection;
    }
    /**
     * Pointwise Linear projection of the values.
     *
     * @return Pointwise Linear projection of the values.
     */
    public Linear getValueProjection() {
        return valueProjection;
    }
    /**
     * Pointwise Linear projection of the results.
     *
     * @return Pointwise Linear projection of the results.
     */
    public Linear getResultProjection() {
        return resultProjection;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        // Return shape is the shape of the query. For 2 or less inputs we have self-attention, i.e.
        // the shape of the output is the shape of the input
        if (inputShapes.length == 1 || inputShapes.length == 2) {
            return new Shape[] {inputShapes[0]};
        } else if (inputShapes.length == 3 || inputShapes.length == 4) {
            // For attention with a dedicated query, the output shape is the query shape
            return new Shape[] {inputShapes[1]};
        } else {
            throw new IllegalArgumentException(
                    "Invalid number of input shapes: " + inputShapes.length + ", must be 1-4.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        // The lookups are fed reshaped input where the batch size is combined with the sequence
        // length.
        // The linear layers only care about the 2nd dimension, so we set the first to -1.
        Shape projectionShape = new Shape(-1L, embeddingSize);
        // We initialize the lookup with that reshaped input shape
        for (Block projection : children.values()) {
            projection.initialize(manager, DataType.FLOAT32, projectionShape);
        }
    }

    /**
     * Utility function to reshape and transpose an input of the shape (B, S, E) into (B, N, S, H).
     *
     * @param projection projected embeddings
     * @param B batch size
     * @param S sequence length
     * @param N number of attention heads
     * @param H size of attention heads
     * @return the reshaped input
     */
    private NDArray createAttentionHeadsFromEmbeddings(
            NDArray projection, long B, long S, long N, long H) {
        // Reshape projection to sequence & heads: (B, S, E) -> (B, S, N, H)
        NDArray sequenceAndHeads = projection.reshape(B, S, N, H);
        // Switch sequence idx & head index, so we have sequences of heads at the end
        return sequenceAndHeads.transpose(0, 2, 1, 3);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        // E=embedding size
        long E = embeddingSize;
        // B=batch size
        long B = inputs.head().getShape().get(0);
        // N=number of attention heads
        long N = headCount;
        // F=from sequence length
        long F;
        // T=to sequence length
        long T;
        // H=Attention head size (= E / N)
        long H = E / N;
        // Create key, query & value input based on input size
        NDList flattenedKeyInput;
        NDList flattenedQueryInput;
        NDList flattenedValueInput;
        NDArray attentionMask;
        if (inputs.size() < 3) { // self attention, either masked or unmasked
            F = inputs.head().getShape().get(1);
            T = F;
            flattenedKeyInput = new NDList(inputs.head());
            flattenedQueryInput = flattenedKeyInput;
            flattenedValueInput = flattenedKeyInput;
        } else { // attention with separate key, query & value
            F = inputs.get(0).getShape().get(1);
            T = inputs.get(1).getShape().get(1);
            flattenedKeyInput = new NDList(inputs.get(0));
            flattenedQueryInput = new NDList(inputs.get(1));
            flattenedValueInput = new NDList(inputs.get(2));
        }
        if (inputs.size() == 2 || inputs.size() == 4) { // we have an additional attention mask
            attentionMask = inputs.get(inputs.size() - 1);
        } else {
            attentionMask = null;
        }
        // apply projection for key, query and value, preserves shape: (B, S, E)
        NDList keys = keyProjection.forward(parameterStore, flattenedKeyInput, training, params);
        NDList queries =
                queryProjection.forward(parameterStore, flattenedQueryInput, training, params);
        NDList values =
                valueProjection.forward(parameterStore, flattenedValueInput, training, params);
        // reshape to (B, N, S, H) to create separate attention heads
        NDArray keyHeads = createAttentionHeadsFromEmbeddings(keys.head(), B, F, N, H);
        NDArray queryHeads = createAttentionHeadsFromEmbeddings(queries.head(), B, T, N, H);
        NDArray valueHeads = createAttentionHeadsFromEmbeddings(values.head(), B, F, N, H);
        // Apply attention by multiplying the key and query vectors: (B, N, T, F)
        // (For each entry in the sequence there is a weight for each other head in the sequence)
        NDArray attentionScores = queryHeads.matMul(keyHeads.transpose(0, 1, 3, 2));
        // Normalize the scores with 1/sqrt(H)
        NDArray normalizedAttentionScores =
                attentionScores.mul(attentionScores.getManager().create(1f / (float) Math.sqrt(H)));
        // Apply masking if requested, mask has shape (B, T, F)
        if (attentionMask != null) {
            NDArray maskOffset;

            // The input mask is initially given as a list of integers with a 1 for each existing
            // token. In order to apply it to the attention result, it needs to be expanded and the
            // values turned into offsets for the softmax calculation. For stacked models, this
            // can be done once and reused - hence we check for the number of dimensions if we
            // have to do this locally or whether it was done for us.
            if (attentionMask.getShape().dimension() != 4) {
                // expand mask to be used on all heads at once
                NDArray expandedMask = attentionMask.reshape(B, 1, T, F);
                // we turn the mask from ints into floats and turn all 1s into 0s and all
                // 0s int o a value of -10000. Adding this to the scores will push all unwanted
                // values towards -inf and keep the unmasked values unchanged
                maskOffset =
                        expandedMask
                                .toType(DataType.FLOAT32, false)
                                .mul(expandedMask.getManager().create(-1f)) // turn 1 into -1
                                .add(
                                        expandedMask
                                                .getManager()
                                                .create(1f)) // turn 0s to 1s, -1s to 0s
                                .mul(
                                        expandedMask
                                                .getManager()
                                                .create(-100000f)); // turn 1s (original 0s) into
                // -100000
            } else {
                maskOffset = attentionMask;
            }
            // adding the mask to the scores removes the scores of unwanted positions
            normalizedAttentionScores = normalizedAttentionScores.add(maskOffset);
        }
        // Then apply softmax to get a probability distribution, shape (B, N, T, F)
        NDArray attentionProbs = normalizedAttentionScores.softmax(3);
        // We apply dropout to the attention probabilities - this will remove entire tokens from the
        // result of a position, as their probability will be set to 0
        NDArray attentionProbsAfterDropout =
                attentionProbsDropout
                        .forward(parameterStore, new NDList(attentionProbs), training)
                        .singletonOrThrow();
        // The result of the attention mechanism is created by a weighted sum using the attention
        // probs. The new head is the weighted sum of the value heads. (B, N, T, H)
        NDArray attentionResult = attentionProbsAfterDropout.matMul(valueHeads);
        // Finally, the heads are reshaped and concatenated into an embedding again
        NDArray resultEmbeddings =
                attentionResult // (B, N, T, H)
                        .transpose(0, 2, 1, 3) // -> (B, T, N, H)
                        .reshape(B, T, E); // -> (B, T, E)
        // As a last step, we add another linear projection for each token to the embedding size
        NDList projectedEmbeddings =
                resultProjection.forward(parameterStore, new NDList(resultEmbeddings), training);
        // done!
        return new NDList(projectedEmbeddings);
    }

    /**
     * Creates a new Builder to build an Attention Block with.
     *
     * @return a new Builder to build an Attention Block with.
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder for {@link ScaledDotProductAttentionBlock}s. */
    public static final class Builder {

        private int embeddingSize;

        private int headCount;

        private float attentionProbsDropoutProb = 0.1f;

        private Builder() {}

        /**
         * Sets the embedding Size to be used for the internal token representation.
         *
         * @param embeddingSize the embedding Size to be used for the internal token representation.
         * @return this builder
         */
        public Builder setEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        /**
         * Sets the number of attention Heads, must divide the embedding size without rest. I.e. if
         * embeddingSize = 10, a headCount of 3 would not be valid, a headCount of 1, 2 or 5 would
         * be.
         *
         * @param headCount the number of attention Heads
         * @return this builder
         */
        public Builder setHeadCount(int headCount) {
            this.headCount = headCount;
            return this;
        }

        /**
         * Sets the probability of applying dropout to the attention probability distribution. This
         * dropout can randomly remove a complete token from the result at a position.
         *
         * @param attentionProbsDropoutProb the probability of applying dropout to the attention
         *     probability distribution
         * @return this builder
         */
        public Builder optAttentionProbsDropoutProb(float attentionProbsDropoutProb) {
            this.attentionProbsDropoutProb = attentionProbsDropoutProb;
            return this;
        }

        /**
         * Creates a new {@code ScaledDotProductAttentionBlock} with the current configuration.
         *
         * @return a new {@code ScaledDotProductAttentionBlock} with the current configuration.
         */
        public ScaledDotProductAttentionBlock build() {
            if (embeddingSize < 1) {
                throw new IllegalStateException("Embedding size not initialized.");
            }
            if (headCount < 1) {
                throw new IllegalStateException("Head count not initialized.");
            }
            if (embeddingSize % headCount != 0) {
                throw new IllegalStateException(
                        "Embedding Size ("
                                + embeddingSize
                                + ") is not divisible by head count ("
                                + headCount
                                + ")");
            }
            return new ScaledDotProductAttentionBlock(this);
        }
    }
}
