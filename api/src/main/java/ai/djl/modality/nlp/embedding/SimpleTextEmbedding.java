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
import ai.djl.ndarray.NDManager;
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
    public NDArray preprocessTextToEmbed(NDManager manager, List<String> text) {
        NDList result = new NDList(text.size());
        for (String token : text) {
            result.add(wordEmbedding.preprocessWordToEmbed(manager, token));
        }
        return NDArrays.stack(result);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedText(NDArray text) throws EmbeddingException {
        NDList split = text.split(text.getShape().get(0));
        NDList result = new NDList();
        for (NDArray token : split) {
            result.add(wordEmbedding.embedWord(token.get(0)));
        }
        return NDArrays.stack(result);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> unembedText(NDArray textEmbedding) throws EmbeddingException {
        NDList split = textEmbedding.split(textEmbedding.getShape().get(0));
        List<String> result = new ArrayList<>(split.size());
        for (NDArray token : split) {
            result.add(wordEmbedding.unembedWord(token.get(0)));
        }
        return result;
    }
}
