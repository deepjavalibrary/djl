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
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Implements the core bert model (without next sentence and masked language task) of bert.
 *
 * <p>This closely follows the original <a href="https://arxiv.org/abs/1810.04805">Devlin et.
 * al.</a> paper and its reference implementation.
 */
// We name local variables for tensor dimensions as in the paper and the reference code.
// While against the general code style, it makes things much easier readable here.
@SuppressWarnings({
    "LocalFinalVariableName",
    "PMD.LocalVariableNamingConventions",
    "ParameterName",
    "PMD.FormalParameterNamingConventions"
})
public final class BertBlock extends AbstractBlock {
    private static final byte VERSION = 1;
    private static final String PARAM_POSITION_EMBEDDING = "positionEmbedding";

    private final int embeddingSize;
    private final int tokenDictionarySize;
    private final int typeDictionarySize;

    private final IdEmbedding tokenEmbedding;
    private final IdEmbedding typeEmbedding;
    private final Parameter positionEmebdding;
    private final BatchNorm embeddingNorm;
    private final Dropout embeddingDropout;
    private final List<TransformerEncoderBlock> transformerEncoderBlocks;
    private final Linear pooling;

    private BertBlock(Builder builder) {
        super(VERSION);
        this.embeddingSize = builder.embeddingSize;
        // embedding for the input tokens
        this.tokenEmbedding =
                addChildBlock(
                        "tokenEmbedding",
                        new IdEmbedding.Builder()
                                .setEmbeddingSize(builder.embeddingSize)
                                .setDictionarySize(builder.tokenDictionarySize)
                                .build());
        this.tokenDictionarySize = builder.tokenDictionarySize;
        // embedding for the position
        this.positionEmebdding =
                addParameter(
                        new Parameter(PARAM_POSITION_EMBEDDING, this, ParameterType.WEIGHT),
                        new Shape(builder.maxSequenceLength, builder.embeddingSize));
        // embedding for the input types
        this.typeEmbedding =
                addChildBlock(
                        "typeEmbedding",
                        new IdEmbedding.Builder()
                                .setEmbeddingSize(builder.embeddingSize)
                                .setDictionarySize(builder.typeDictionarySize)
                                .build());
        this.typeDictionarySize = builder.typeDictionarySize;
        // normalizer for the embeddings
        this.embeddingNorm = addChildBlock("embeddingNorm", BatchNorm.builder().optAxis(2).build());
        // dropout to apply after embedding normalization
        this.embeddingDropout =
                addChildBlock(
                        "embeddingDropout",
                        Dropout.builder().optProbability(builder.hiddenDropoutProbability).build());
        // the transformer blocks
        this.transformerEncoderBlocks = new ArrayList<>(builder.transformerBlockCount);
        for (int i = 0; i < builder.transformerBlockCount; ++i) {
            this.transformerEncoderBlocks.add(
                    addChildBlock(
                            "transformer_" + i,
                            new TransformerEncoderBlock(
                                    builder.embeddingSize,
                                    builder.attentionHeadCount,
                                    builder.hiddenSize,
                                    0.1f,
                                    Activation::gelu)));
        }
        // add projection for pooling layer
        this.pooling =
                addChildBlock(
                        "poolingProjection",
                        Linear.builder()
                                .setOutChannels(builder.embeddingSize)
                                .optBias(true)
                                .build());
    }

    /**
     * Returns the token embedding used by this Bert model.
     *
     * @return the token embedding used by this Bert model
     */
    public IdEmbedding getTokenEmbedding() {
        return this.tokenEmbedding;
    }

    /**
     * Returns the embedding size used for tokens.
     *
     * @return the embedding size used for tokens
     */
    public int getEmbeddingSize() {
        return embeddingSize;
    }

    /**
     * Returns the size of the token dictionary.
     *
     * @return the size of the token dictionary
     */
    public int getTokenDictionarySize() {
        return tokenDictionarySize;
    }

    /**
     * Returns the size of the type dictionary.
     *
     * @return the size of the type dictionary
     */
    public int getTypeDictionarySize() {
        return typeDictionarySize;
    }

    /**
     * Gets the output shapes.
     *
     * @param inputShapes tokenIds (B, S), typeIds (B, S), masks (B, S)
     * @return embeddings (B, S, E), pooled classifier (B, E)
     */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        final long B = inputShapes[0].get(0);
        final long S = inputShapes[0].get(1);
        return new Shape[] {new Shape(B, S, embeddingSize), new Shape(B, embeddingSize)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        inputNames = Arrays.asList("tokenIds", "typeIds", "masks");
        Shape[] tokenShape = {inputShapes[0]};
        Shape[] typeShape = {inputShapes[1]};
        Shape[] embeddingOutput = this.tokenEmbedding.initialize(manager, dataType, tokenShape);
        this.typeEmbedding.initialize(manager, dataType, typeShape);
        this.embeddingNorm.initialize(manager, dataType, embeddingOutput);
        this.embeddingDropout.initialize(manager, dataType, embeddingOutput);
        for (final TransformerEncoderBlock tb : transformerEncoderBlocks) {
            tb.initialize(manager, dataType, embeddingOutput);
        }
        long batchSize = inputShapes[0].get(0);
        this.pooling.initialize(manager, dataType, new Shape(batchSize, embeddingSize));
    }

    /**
     * Creates a 3D attention mask from a 2D tensor mask.
     *
     * @param ids 2D Tensor of shape (B, F)
     * @param mask 2D Tensor of shape (B, T)
     * @return float tensor of shape (B, F, T)
     */
    public static NDArray createAttentionMaskFromInputMask(NDArray ids, NDArray mask) {
        long batchSize = ids.getShape().get(0);
        long fromSeqLength = ids.getShape().get(1);
        long toSeqLength = mask.getShape().get(1);
        // we ignore the actual content of the ids, we just create a "pseudo-mask" of ones for them
        NDArray broadcastOnes =
                ids.onesLike().toType(DataType.FLOAT32, false).reshape(batchSize, fromSeqLength, 1);
        // add empty dimension to multiply with broadcasted ones
        NDArray mask3D = mask.toType(DataType.FLOAT32, false).reshape(batchSize, 1, toSeqLength);

        return broadcastOnes.matMul(mask3D);
    }

    @Override
    public NDList forward(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        return forward(ps, inputs, training);
    }

    @Override
    public NDList forward(final ParameterStore ps, final NDList inputs, boolean training) {
        // First input are the tokens.
        NDArray tokenIds = inputs.get(0);
        // Second are the token types (first / second sentence).
        NDArray typeIds = inputs.get(1);
        // Third are the masks for the input
        NDArray masks = inputs.get(2);
        return forward(ps, tokenIds, typeIds, masks, training);
    }

    /**
     * Embeds the input, runs it through a transformer encoder stack and outputs the resulting
     * embeddings and the pooled classifier.
     *
     * @param ps the parameter store
     * @param tokenIds Ids for the tokens (word pieces, BPE pieces, words etc.) (B, S)
     * @param typeIds Ids for the token type, during pretraining this is used to encode the
     *     first/second sentence (B, S)
     * @param masks Masks for the input, used to mask out tokens in input that is shorter than the
     *     max number of input tokens (B, S)
     * @param training true=apply dropout etc.
     * @return token embeddings (B, S, E), pooled classifier (B, E)
     */
    public NDList forward(
            ParameterStore ps, NDArray tokenIds, NDArray typeIds, NDArray masks, boolean training) {
        MemoryScope initScope = MemoryScope.from(tokenIds).add(typeIds, masks);
        // Create embeddings for inputs
        NDArray embeddedTokens = tokenEmbedding.forward(ps, tokenIds);
        NDArray embeddedTypes = typeEmbedding.forward(ps, typeIds);
        NDArray embeddedPositions = ps.getValue(positionEmebdding, tokenIds.getDevice());
        // Merge them to one embedding by adding them
        // (We can just add the position embedding, even though it does not have a batch dimension:
        // the tensor is automagically "broadcast" i.e. repeated in the batch dimension. That
        // gives us the result we want: every embedding gets the same position embedding added
        // to it)
        NDArray embedding = embeddedTokens.add(embeddedTypes).add(embeddedPositions);
        // Apply normalization
        NDList normalizedEmbedding = embeddingNorm.forward(ps, new NDList(embedding), training);
        NDList dropoutEmbedding = embeddingDropout.forward(ps, normalizedEmbedding, training);
        // create 3D attention mask
        NDArray attentionMask = createAttentionMaskFromInputMask(tokenIds, masks);
        Shape maskShape = attentionMask.getShape();
        NDArray offsetMask =
                attentionMask
                        .reshape(maskShape.get(0), 1, maskShape.get(1), maskShape.get(2))
                        .toType(DataType.FLOAT32, false)
                        .mul(-1f) // turn 1 into -1
                        .add(1f) // turn 0s to 1s, -1s to 0s
                        .mul(-100000f); // turn 1s (original 0s) into -100000
        // Run through all transformer blocks
        NDList lastOutput = dropoutEmbedding;
        initScope
                .remove(tokenIds, typeIds, masks)
                .waitToRead(dropoutEmbedding)
                .waitToRead(offsetMask)
                .close();
        for (final TransformerEncoderBlock block : transformerEncoderBlocks) {
            NDList input = new NDList(lastOutput.head(), offsetMask);
            MemoryScope innerScope = MemoryScope.from(input);
            lastOutput = block.forward(ps, input, training);
            innerScope.remove(offsetMask).waitToRead(lastOutput).close();
        }
        // We also return the pooled output - this is an additional fully connected layer
        // only applied to the first token, assumed to be the CLS token to be used for training
        // classifiers. shape = (B, E) We apply a tanh activation to it.
        NDArray firstToken = lastOutput.head().get(new NDIndex(":,1,:")).squeeze();
        NDArray pooledFirstToken =
                pooling.forward(ps, new NDList(firstToken), training).head().tanh();
        lastOutput.add(pooledFirstToken);
        return lastOutput;
    }

    /**
     * Returns a new BertBlock builder.
     *
     * @return a new BertBlock builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link BertBlock} type of {@link Block}. */
    public static final class Builder {
        int tokenDictionarySize;
        int typeDictionarySize = 16;
        int embeddingSize = 768;
        int transformerBlockCount = 12;
        int attentionHeadCount = 12;
        int hiddenSize = 4 * embeddingSize;
        float hiddenDropoutProbability = 0.1f;
        // float attentionDropoutProbability = 0.1f;
        int maxSequenceLength = 512;
        // float initializerRange = 0.02f;

        private Builder() {}

        /**
         * Sets the number of tokens in the dictionary.
         *
         * @param tokenDictionarySize the number of tokens in the dictionary
         * @return this builder
         */
        public Builder setTokenDictionarySize(int tokenDictionarySize) {
            this.tokenDictionarySize = tokenDictionarySize;
            return this;
        }

        /**
         * Sets the number of possible token types. This should be a very small number (2-16).
         *
         * @param typeDictionarySize the number of possible token types. This should be a very small
         *     number (2-16)
         * @return this builder
         */
        public Builder optTypeDictionarySize(int typeDictionarySize) {
            this.typeDictionarySize = typeDictionarySize;
            return this;
        }

        /**
         * Sets the embedding size to use for input tokens. This size must be divisible by the
         * number of attention heads.
         *
         * @param embeddingSize the embedding size to use for input tokens.
         * @return this builder
         */
        public Builder optEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        /**
         * Sets the number of transformer blocks to use.
         *
         * @param transformerBlockCount the number of transformer blocks to use
         * @return this builder
         */
        public Builder optTransformerBlockCount(int transformerBlockCount) {
            this.transformerBlockCount = transformerBlockCount;
            return this;
        }

        /**
         * Sets the number of attention heads to use in each transformer block. This number must
         * divide the embedding size without rest.
         *
         * @param attentionHeadCount the number of attention heads to use in each transformer block.
         * @return this builder
         */
        public Builder optAttentionHeadCount(int attentionHeadCount) {
            this.attentionHeadCount = attentionHeadCount;
            return this;
        }

        /**
         * Sets the size of the hidden layers in the fully connected networks used.
         *
         * @param hiddenSize the size of the hidden layers in the fully connected networks used.
         * @return this builder
         */
        public Builder optHiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        /**
         * Sets the dropout probabilty in the hidden fully connected networks.
         *
         * @param hiddenDropoutProbability the dropout probabilty in the hidden fully connected
         *     networks.
         * @return this builder
         */
        public Builder optHiddenDropoutProbability(float hiddenDropoutProbability) {
            this.hiddenDropoutProbability = hiddenDropoutProbability;
            return this;
        }

        /**
         * Sets the probabilty for attention dropout.
         *
         * @param attentionDropoutProbability the probabilty for attention dropout
         * @return this builder
         */
        /*public Builder optAttentionDropoutProbability(float attentionDropoutProbability) {
            this.attentionDropoutProbability = attentionDropoutProbability;
            return this;
        }*/

        /**
         * Sets the maximum sequence length this model can process. Memory & compute requirements of
         * the attention mechanism is O(n²), so large values can easily exhaust your GPU memory!
         *
         * @param maxSequenceLength the maximum sequence length this model can process.
         * @return this builder
         */
        public Builder optMaxSequenceLength(int maxSequenceLength) {
            this.maxSequenceLength = maxSequenceLength;
            return this;
        }

        /**
         * Sets the initializer range for the truncated normal initialization used for this model.
         *
         * @param initializerRange the initializer range for the truncated normal initialization
         *     used for this model.
         * @return this builder
         */
        /*public Builder optInitializerRange(float initializerRange) {
            this.initializerRange = initializerRange;
            return this;
        }*/

        /**
         * Tiny config for testing on laptops.
         *
         * @return this builder
         */
        public Builder nano() {
            typeDictionarySize = 2;
            embeddingSize = 256;
            transformerBlockCount = 4;
            attentionHeadCount = 4;
            hiddenSize = 4 * embeddingSize;
            hiddenDropoutProbability = 0.1f;
            // attentionDropoutProbability = 0.1f;
            maxSequenceLength = 128;
            // initializerRange = 0.02f;
            return this;
        }

        /**
         * Sets this builder's params to a minimal configuration that nevertheless performs quite
         * well.
         *
         * @return this builder
         */
        public Builder micro() {
            typeDictionarySize = 2;
            embeddingSize = 512;
            transformerBlockCount = 12;
            attentionHeadCount = 8;
            hiddenSize = 4 * embeddingSize;
            hiddenDropoutProbability = 0.1f;
            // attentionDropoutProbability = 0.1f;
            maxSequenceLength = 128;
            // initializerRange = 0.02f;
            return this;
        }

        /**
         * Sets this builder's params to the BASE config of the original BERT paper. (except for the
         * dictionary size)
         *
         * @return this builder
         */
        public Builder base() {
            typeDictionarySize = 16;
            embeddingSize = 768;
            transformerBlockCount = 12;
            attentionHeadCount = 12;
            hiddenSize = 4 * embeddingSize;
            hiddenDropoutProbability = 0.1f;
            // attentionDropoutProbability = 0.1f;
            maxSequenceLength = 256;
            // initializerRange = 0.02f;
            return this;
        }

        /**
         * Sets this builder's params to the LARGE config of the original BERT paper. (except for
         * the dictionary size)
         *
         * @return this builder
         */
        public Builder large() {
            typeDictionarySize = 16;
            embeddingSize = 1024;
            transformerBlockCount = 24;
            attentionHeadCount = 16;
            hiddenSize = 4 * embeddingSize;
            hiddenDropoutProbability = 0.1f;
            // attentionDropoutProbability = 0.1f;
            maxSequenceLength = 512;
            // initializerRange = 0.02f;
            return this;
        }

        /**
         * Returns a new BertBlock with the parameters of this builder.
         *
         * @return a new BertBlock with the parameters of this builder.
         */
        public BertBlock build() {
            if (tokenDictionarySize == 0) {
                throw new IllegalArgumentException("You must specify the dictionary size.");
            }
            return new BertBlock(this);
        }
    }
}
