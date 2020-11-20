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
import ai.djl.nn.core.Embedding;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import java.util.List;

/** A {@link WordEmbedding} using a {@link ZooModel}. */
public class ModelZooTextEmbedding implements TextEmbedding, AutoCloseable {

    private Predictor<NDList, NDList> predictor;
    private Embedding<String> embedding;

    /**
     * Constructs a {@link ModelZooTextEmbedding}.
     *
     * @param model the model for the embedding. The model's block must consist of only an {@link
     *     Embedding}&lt;{@link String}&gt;.
     */
    @SuppressWarnings("unchecked")
    public ModelZooTextEmbedding(Model model) {
        predictor = model.newPredictor(new NoopTranslator(Batchifier.STACK));
        try {
            embedding = (Embedding<String>) model.getBlock();
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("The model was not an embedding", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public long[] preprocessTextToEmbed(List<String> tokens) {
        return tokens.stream().mapToLong(token -> embedding.embed(token)).toArray();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedText(NDArray indices) throws EmbeddingException {
        try {
            return predictor.predict(new NDList(indices)).singletonOrThrow();
        } catch (TranslateException e) {
            throw new EmbeddingException("Could not embed word", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<String> unembedText(NDArray word) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        predictor.close();
    }
}
