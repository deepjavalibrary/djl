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
package ai.djl.modality.nlp.embedding;

import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.core.Embedding;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code TrainableWordEmbedding} is an implementation of {@link WordEmbedding} and {@link
 * Embedding} based on a {@link SimpleVocabulary}. This {@link WordEmbedding} is ideal when there
 * are no pre-trained embeddings available.
 */
public class TrainableWordEmbedding extends Embedding<String> implements WordEmbedding {
    private static final String DEFAULT_UNKNOWN_TOKEN = "<unk>";

    private String unknownToken;

    /**
     * Constructs a new instance of {@code TrainableWordEmbedding} from the {@link Builder}.
     *
     * @param builder the {@link Builder}
     */
    public TrainableWordEmbedding(Builder builder) {
        super(builder);
        this.unknownToken = builder.unknownToken;
    }

    /**
     * Constructs a new instance of {@code TrainableWordEmbedding} from a {@link SimpleVocabulary}
     * and a given embedding size.
     *
     * @param simpleVocabulary a {@link SimpleVocabulary} to get tokens from
     * @param embeddingSize the required embedding size
     */
    public TrainableWordEmbedding(SimpleVocabulary simpleVocabulary, int embeddingSize) {
        super(
                builder()
                        .setEmbeddingSize(embeddingSize)
                        .setItems(new ArrayList<>(simpleVocabulary.getAllTokens())));
        this.unknownToken = simpleVocabulary.getUnknownToken();
    }

    /**
     * Constructs a pretrained embedding.
     *
     * @param embedding the embedding array
     * @param items the items in the embedding (in matching order to the embedding array)
     */
    public TrainableWordEmbedding(NDArray embedding, List<String> items) {
        super(embedding, items);
        this.unknownToken = DEFAULT_UNKNOWN_TOKEN;
    }

    /**
     * Constructs a pretrained embedding.
     *
     * @param embedding the embedding array
     * @param items the items in the embedding (in matching order to the embedding array)
     * @param sparseGrad whether to compute row sparse gradient in the backward calculation
     */
    public TrainableWordEmbedding(NDArray embedding, List<String> items, boolean sparseGrad) {
        super(embedding, items, sparseGrad);
        this.unknownToken = DEFAULT_UNKNOWN_TOKEN;
    }

    /** {@inheritDoc} */
    @Override
    public boolean vocabularyContains(String word) {
        return embedder.containsKey(word);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray preprocessWordToEmbed(NDManager manager, String word) {
        if (hasItem(word)) {
            return embed(manager, word);
        }
        return embed(manager, unknownToken);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedWord(NDArray word) {
        throw new UnsupportedOperationException("This operation is not supported by this class.");
    }

    /** {@inheritDoc} */
    @Override
    public String unembedWord(NDArray word) throws EmbeddingException {
        if (!word.isScalar()) {
            throw new EmbeddingException("NDArray word must be scalar index");
        }
        return unembed(word.toIntArray()[0]).orElseGet(() -> unknownToken);
    }

    /** {@inheritDoc} */
    @Override
    public byte[] encode(String input) {
        byte[] encodedInput;
        encodedInput = input.getBytes(StandardCharsets.UTF_8);
        return encodedInput;
    }

    /** {@inheritDoc} */
    @Override
    public String decode(byte[] byteArray) {
        return new String(byteArray, StandardCharsets.UTF_8);
    }

    /**
     * Creates a builder to build an {@link Embedding}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new TrainableWordEmbedding.Builder();
    }

    /** A builder for a {@link TrainableWordEmbedding}. */
    public static class Builder extends Embedding.BaseBuilder<String, Builder> {
        private String unknownToken = DEFAULT_UNKNOWN_TOKEN;

        Builder() {
            super();
            this.embeddingType = String.class;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder setType(Class<String> embeddingType) {
            return self();
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the optional {@link String} value for the unknown token.
         *
         * @param unknownToken the {@link String} value of unknown token
         * @return this Builder
         */
        public Builder optUnknownToken(String unknownToken) {
            this.unknownToken = unknownToken;
            return self();
        }

        /**
         * Builds a new instance of {@link TrainableWordEmbedding} based on the arguments in this
         * builder.
         *
         * @return a new instance of {@link TrainableWordEmbedding}
         */
        public TrainableWordEmbedding build() {
            return new TrainableWordEmbedding(this);
        }
    }
}
