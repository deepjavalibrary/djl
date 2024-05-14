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
package ai.djl.sentencepiece;

import ai.djl.modality.nlp.Vocabulary;

/** {@link SpVocabulary} is a SentencePiece implementation of {@link Vocabulary}. */
public final class SpVocabulary implements Vocabulary {

    private SpProcessor processor;

    // TODO: Support direct Vocabulary loading
    private SpVocabulary(SpProcessor processor) {
        this.processor = processor;
    }

    /**
     * Get Vocabulary from {@link SpTokenizer}.
     *
     * @param tokenizer the {@link SpTokenizer}
     * @return {@link SpVocabulary}
     */
    public static SpVocabulary from(SpTokenizer tokenizer) {
        return new SpVocabulary(tokenizer.getProcessor());
    }

    /** {@inheritDoc} */
    @Override
    public String getToken(long index) {
        return processor.getToken((int) index);
    }

    /** {@inheritDoc} */
    @Override
    public boolean contains(String token) {
        throw new UnsupportedOperationException("Not supported for Sentence Piece");
    }

    /** {@inheritDoc} */
    @Override
    public long getIndex(String token) {
        return processor.getId(token);
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        throw new UnsupportedOperationException("Not supported for Sentence Piece");
    }
}
