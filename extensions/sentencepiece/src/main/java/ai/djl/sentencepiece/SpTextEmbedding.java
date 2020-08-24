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

import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.ndarray.NDArray;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A {@link TextEmbedding} in SentencePiece that do sentence tokenization and map tokens into
 * indices.
 */
public final class SpTextEmbedding implements TextEmbedding {

    private SpProcessor processor;

    private SpTextEmbedding(SpProcessor processor) {
        this.processor = processor;
    }

    /**
     * Get SentencePiece TextEmbeeding from {@link SpTokenizer}.
     *
     * @param tokenizer the {@link SpTokenizer}
     * @return {@link SpTextEmbedding}
     */
    public static SpTextEmbedding from(SpTokenizer tokenizer) {
        return new SpTextEmbedding(tokenizer.getProcessor());
    }

    /** {@inheritDoc} */
    @Override
    public long[] preprocessTextToEmbed(List<String> text) {
        if (text.size() != 1) {
            throw new IllegalArgumentException(
                    "SentencePiece require one single sentence to be passed as text");
        }
        int[] indices = processor.encode(text.get(0));
        return Arrays.stream(indices).asLongStream().toArray();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedText(NDArray textIndices) {
        return textIndices;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> unembedText(NDArray textEmbedding) {
        long[] indices = textEmbedding.toLongArray();
        String result = processor.decode(Arrays.stream(indices).mapToInt(i -> (int) i).toArray());
        return Collections.singletonList(result);
    }
}
