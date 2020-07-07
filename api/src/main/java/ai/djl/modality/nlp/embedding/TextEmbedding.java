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
import ai.djl.ndarray.NDManager;
import java.util.List;

/**
 * A class to manage 1-D {@link NDArray} representations of multiple words.
 *
 * <p>A text embedding differs from a {@link ai.djl.modality.nlp.embedding.WordEmbedding} because
 * the text embedding does not have to be applied to each word independently.
 *
 * <p>A text embedding maps text to a {@link NDArray} that attempts to represent the key ideas in
 * the words. Each of the values in the dimension can represent different pieces of meaning such as
 * young-old, object-living, etc.
 *
 * <p>These text embeddings can be used in two different ways in models. First, they can be used
 * purely for preprocessing the model. In this case, it is a requirement for most models that use
 * text as an input. The model is not trained. For this use case, use {@link #embedText}.
 *
 * <p>In the second option, the embedding can be trained using the standard deep learning techniques
 * to better handle the current dataset. For this case, you need two methods. First, call {@link
 * #preprocessTextToEmbed(List)} within your dataset. Then, the first step in your model should be
 * to call {@link #embedText(NDManager, long[])}.
 */
public interface TextEmbedding {

    /**
     * Preprocesses the text to embed into an array to pass into the model.
     *
     * <p>Make sure to call {@link #embedText(NDManager, long[])} after this.
     *
     * @param text the text to embed
     * @return the indices of text that is ready to embed
     */
    long[] preprocessTextToEmbed(List<String> text);

    /**
     * Embeds a text.
     *
     * @param manager the manager for the embedding array
     * @param text the text to embed
     * @return the embedded text
     * @throws EmbeddingException if there is an error while trying to embed
     */
    default NDArray embedText(NDManager manager, List<String> text) throws EmbeddingException {
        return embedText(manager, preprocessTextToEmbed(text));
    }

    /**
     * Embeds the text after preprocessed using {@link #preprocessTextToEmbed(List)}.
     *
     * @param manager the manager to create the embedding array
     * @param textIndices the indices of text to embed
     * @return the embedded text
     * @throws EmbeddingException if there is an error while trying to embed
     */
    default NDArray embedText(NDManager manager, long[] textIndices) throws EmbeddingException {
        return embedText(manager.create(textIndices));
    }

    /**
     * Embeds the text after preprocessed using {@link #preprocessTextToEmbed(List)}.
     *
     * @param textIndices the indices of text to embed
     * @return the embedded text
     * @throws EmbeddingException if there is an error while trying to embed
     */
    NDArray embedText(NDArray textIndices) throws EmbeddingException;

    /**
     * Returns the closest matching text for a given embedding.
     *
     * @param textEmbedding the text embedding to find the matching string text for.
     * @return text similar to the passed in embedding
     * @throws EmbeddingException if the input is not unembeddable
     */
    List<String> unembedText(NDArray textEmbedding) throws EmbeddingException;
}
