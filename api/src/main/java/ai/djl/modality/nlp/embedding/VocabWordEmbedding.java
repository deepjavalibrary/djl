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

import ai.djl.modality.nlp.Vocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.core.Embedding;
import ai.djl.training.ParameterStore;

/**
 * {@code VocabWordEmbedding} is an implementation of {@link WordEmbedding} based on a Vocabulary or
 * an {@link Embedding} block. This {@link WordEmbedding} is ideal when there is no pre-trained
 * embeddings available, or when the pre-trained embedding needs to further trained.
 */
public class VocabWordEmbedding implements WordEmbedding {
    private static final String DEFAULT_UNKNOWN_TOKEN = "<unk>";

    private Embedding<String> embedding;
    private String unknownToken;

    /**
     * Constructs a new instance {@code VocabWordEmbedding} from a given {@link Embedding} block.
     *
     * @param embedding the {@link Embedding} block
     */
    public VocabWordEmbedding(Embedding<String> embedding) {
        this(embedding, DEFAULT_UNKNOWN_TOKEN);
    }

    /**
     * Constructs a new instance {@code VocabWordEmbedding} from a given {@link Embedding} block.
     *
     * @param embedding the {@link Embedding} block
     * @param unknownToken the {@link String} value of unknown token
     */
    public VocabWordEmbedding(Embedding<String> embedding, String unknownToken) {
        this.embedding = embedding;
        this.unknownToken = unknownToken;
    }

    /**
     * Constructs a new instance {@code VocabWordEmbedding} based on the given {@link Vocabulary}
     * and embedding size.
     *
     * @param vocabulary the {@link Vocabulary} based on which the embedding is built.
     * @param embeddingSize the size of the embedding for each word
     */
    public VocabWordEmbedding(Vocabulary vocabulary, int embeddingSize) {
        this(vocabulary.newEmbedding(embeddingSize), DEFAULT_UNKNOWN_TOKEN);
    }

    /** {@inheritDoc} */
    @Override
    public boolean vocabularyContains(String word) {
        return embedding.hasItem(word);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray preprocessWordToEmbed(NDManager manager, String word) {
        if (embedding.hasItem(word)) {
            return embedding.embed(manager, word);
        }
        return embedding.embed(manager, unknownToken);
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedWord(ParameterStore parameterStore, NDArray word) {
        return embedding.forward(parameterStore, new NDList(word));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedWord(NDArray word) {
        throw new UnsupportedOperationException("This operation is not supported by this class.");
    }

    /** {@inheritDoc} */
    @Override
    public String unembedWord(NDArray wordEmbedding) {
        throw new UnsupportedOperationException("This operation is not supported yet.");
    }
}
