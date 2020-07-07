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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import java.util.ArrayList;
import java.util.List;

/** A {@link TextEmbedding} that applies a {@link WordEmbedding} to each word independently. */
public class SimpleTextEmbedding implements TextEmbedding {

    private WordEmbedding wordEmbedding;

    /**
     * Constructs a {@link SimpleTextEmbedding}.
     *
     * @param wordEmbedding the word embedding to embed each word
     */
    public SimpleTextEmbedding(WordEmbedding wordEmbedding) {
        this.wordEmbedding = wordEmbedding;
    }

    /** {@inheritDoc} */
    @Override
    public long[] preprocessTextToEmbed(List<String> text) {
        long[] result = new long[text.size()];
        for (int i = 0; i < text.size(); i++) {
            result[i] = wordEmbedding.preprocessWordToEmbed(text.get(i));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedText(NDArray textIndices) throws EmbeddingException {
        NDList result = new NDList();
        for (int i = 0; i < textIndices.size(); i++) {
            result.add(wordEmbedding.embedWord(textIndices.get(i)));
        }
        return NDArrays.stack(result);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> unembedText(NDArray textEmbedding) {
        NDList split = textEmbedding.split(textEmbedding.getShape().get(0));
        List<String> result = new ArrayList<>(split.size());
        for (NDArray token : split) {
            result.add(wordEmbedding.unembedWord(token.get(0)));
        }
        return result;
    }
}
