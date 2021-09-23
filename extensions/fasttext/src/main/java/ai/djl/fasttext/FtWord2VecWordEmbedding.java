/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext;

import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.WordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/** An implementation of {@link WordEmbedding} for FastText word embeddings. */
public class FtWord2VecWordEmbedding implements WordEmbedding {

    private FtModel model;
    private Vocabulary vocabulary;

    /**
     * Constructs a {@link FtWord2VecWordEmbedding}.
     *
     * @param model a loaded FastText model
     * @param vocabulary the {@link Vocabulary} to get indices from
     */
    public FtWord2VecWordEmbedding(FtModel model, Vocabulary vocabulary) {
        this.model = model;
        this.vocabulary = vocabulary;
    }

    /** {@inheritDoc} */
    @Override
    public boolean vocabularyContains(String word) {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public long preprocessWordToEmbed(String word) {
        return vocabulary.getIndex(word);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedWord(NDArray index) {
        return embedWord(index.getManager(), index.toLongArray()[0]);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedWord(NDManager manager, long index) {
        String word = vocabulary.getToken(index);
        float[] buf = model.fta.getDataVector(word);
        return manager.create(buf);
    }

    /** {@inheritDoc} */
    @Override
    public String unembedWord(NDArray word) {
        if (!word.isScalar()) {
            throw new IllegalArgumentException("NDArray word must be scalar index");
        }
        return vocabulary.getToken(word.toLongArray()[0]);
    }
}
