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

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.core.Embedding;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Optional;

/**
 * {@code TrainableWordEmbedding} is an implementation of {@link WordEmbedding} and {@link
 * Embedding} based on a {@link DefaultVocabulary}. This {@link WordEmbedding} is ideal when there
 * are no pre-trained embeddings available.
 */
public class TrainableWordEmbedding extends Embedding<String> implements WordEmbedding {

    private static final String DEFAULT_UNKNOWN_TOKEN = "<unk>";

    private Vocabulary vocabulary;

    /**
     * Constructs a new instance of {@code TrainableWordEmbedding} from the {@link Builder}.
     *
     * @param builder the {@link Builder}
     */
    public TrainableWordEmbedding(Builder builder) {
        super(builder);
        this.vocabulary = builder.vocabulary;
    }

    /**
     * Constructs a new instance of {@code TrainableWordEmbedding} from a {@link DefaultVocabulary}
     * and a given embedding size.
     *
     * @param vocabulary a {@link Vocabulary} to get tokens from
     * @param embeddingSize the required embedding size
     */
    public TrainableWordEmbedding(Vocabulary vocabulary, int embeddingSize) {
        super(
                builder()
                        .setEmbeddingSize(embeddingSize)
                        .optDefaultItem(DEFAULT_UNKNOWN_TOKEN)
                        .optUseDefault(false));
        this.vocabulary = vocabulary;
    }

    /**
     * Constructs a pretrained embedding.
     *
     * @param embedding the embedding array
     * @param items the items in the embedding (in matching order to the embedding array)
     */
    public TrainableWordEmbedding(NDArray embedding, List<String> items) {
        super(embedding);
        this.fallthroughEmbedding = new DefaultItem(DEFAULT_UNKNOWN_TOKEN);
        this.vocabulary = new DefaultVocabulary(items);
    }

    /**
     * Constructs a pretrained embedding.
     *
     * @param embedding the embedding array
     * @param items the items in the embedding (in matching order to the embedding array)
     * @param sparseFormat whether to compute row sparse gradient in the backward calculation
     */
    public TrainableWordEmbedding(
            NDArray embedding, List<String> items, SparseFormat sparseFormat) {
        super(embedding, sparseFormat);
        this.fallthroughEmbedding = new DefaultItem(DEFAULT_UNKNOWN_TOKEN);
        this.vocabulary = new DefaultVocabulary(items);
    }

    /** {@inheritDoc} */
    @Override
    public boolean vocabularyContains(String word) {
        return vocabulary.getIndex(word) >= 0;
    }

    /** {@inheritDoc} */
    @Override
    public long preprocessWordToEmbed(String word) {
        return embed(word);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedWord(NDArray index) throws EmbeddingException {
        throw new UnsupportedOperationException(
                "EmbedWord operation is not supported by this class.");
    }

    /** {@inheritDoc} */
    @Override
    public String unembedWord(NDArray word) {
        if (!word.isScalar()) {
            throw new IllegalArgumentException("NDArray word must be scalar index");
        }
        long wordIndex = word.toLongArray()[0];

        Optional<String> result = unembed(wordIndex);
        if (result.isPresent()) {
            return result.get();
        }

        result = fallthroughEmbedding.unembed(wordIndex);
        if (result.isPresent()) {
            return result.get();
        }

        throw new IllegalArgumentException("Failed to unembed word");
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

    /** {@inheritDoc} */
    @Override
    public long embed(String item) {
        if (vocabularyContains(item)) {
            return vocabulary.getIndex(item);
        } else {
            if (fallthroughEmbedding != null) {
                return fallthroughEmbedding.embed(item);
            } else {
                throw new IllegalArgumentException("The provided item was not found");
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public Optional<String> unembed(long index) {
        if (index == -1) {
            if (fallthroughEmbedding == null) {
                throw new IllegalArgumentException(
                        "Index -1 is reserved for the fallThrough but no fallThrough is found");
            }
            return fallthroughEmbedding.unembed(index);
        }
        return Optional.ofNullable(vocabulary.getToken(index));
    }

    /**
     * Creates a builder to build an {@link Embedding}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new TrainableWordEmbedding.Builder();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasItem(String item) {
        return false;
    }

    /** A builder for a {@link TrainableWordEmbedding}. */
    public static class Builder extends Embedding.BaseBuilder<String, Builder> {
        private Vocabulary vocabulary;

        Builder() {
            super();
            this.embeddingType = String.class;
            this.defaultItem = DEFAULT_UNKNOWN_TOKEN;
        }

        /**
         * Sets the {@link Vocabulary} to be used.
         *
         * @param vocabulary the {@link Vocabulary} to be set
         * @return this Builder
         */
        public Builder setVocabulary(Vocabulary vocabulary) {
            this.vocabulary = vocabulary;
            numEmbeddings = Math.toIntExact(vocabulary.size());
            return self();
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
            return optDefaultItem(unknownToken);
        }

        /**
         * Builds a new instance of {@link TrainableWordEmbedding} based on the arguments in this
         * builder.
         *
         * @return a new instance of {@link TrainableWordEmbedding}
         */
        public TrainableWordEmbedding build() {
            if (numEmbeddings != vocabulary.size()) {
                throw new IllegalArgumentException(
                        "The numEmbeddings is "
                                + numEmbeddings
                                + " and the vocabulary has size "
                                + vocabulary.size()
                                + " but they should be equal.");
            }
            return new TrainableWordEmbedding(this);
        }
    }
}
