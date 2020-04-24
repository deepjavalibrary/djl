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

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.core.Embedding;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

/** A {@link WordEmbedding} using a {@link ZooModel}. */
public class ModelZooWordEmbedding implements WordEmbedding, AutoCloseable {

    private Predictor<NDList, NDList> predictor;
    private Embedding<String> embedding;
    private String unknownToken;

    /**
     * Constructs a {@link ModelZooWordEmbedding}.
     *
     * @param model the model for the embedding. The model's block must consist of only an {@link
     *     Embedding}&lt;{@link String}&gt;.
     */
    @SuppressWarnings("unchecked")
    public ModelZooWordEmbedding(Model model) {
        this.unknownToken = model.getProperty("unknownToken");
        predictor = model.newPredictor(new NoopTranslator());
        try {
            embedding = (Embedding<String>) model.getBlock();
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("The model was not an embedding", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean vocabularyContains(String word) {
        return embedding.hasItem(word);
    }

    /** {@inheritDoc} */
    @Override
    public int preprocessWordToEmbed(String word) {
        if (embedding.hasItem(word)) {
            return embedding.embed(word);
        }
        return embedding.embed(unknownToken);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedWord(NDManager manager, int word) throws EmbeddingException {
        try {
            return predictor.predict(new NDList(manager.create(word))).singletonOrThrow();
        } catch (TranslateException e) {
            throw new EmbeddingException("Could not embed word", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String unembedWord(NDArray word) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        predictor.close();
    }
}
