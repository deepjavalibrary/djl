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

/**
 * A class to manage 1-D {@link NDArray} representations of words.
 *
 * <p>A word embedding maps words to a {@link NDArray} that attempts to represent the key ideas in
 * the words. Each of the values in the dimension can represent different pieces of meaning such as
 * young-old, object-living, etc.
 *
 * <p>These word embeddings can be used in two different ways in models. First, they can be used
 * purely for preprocessing the model. In this case, it is a requirement for most models that use
 * text as an input. The model is not trained. For this use case, use {@link #embedWord}.
 *
 * <p>In the second option, the embedding can be trained using the standard deep learning techniques
 * to better handle the current dataset. For this case, you need two methods. First, call {@link
 * #preprocessWordToEmbed(NDManager, String)} within your dataset. Then, the first step in your
 * model should be to call {@link #embedWord(NDArray)}.
 */
public interface WordEmbedding {

    /**
     * Returns whether an embedding exists for a word.
     *
     * @param word the word to check
     * @return true if an embedding exists
     */
    boolean vocabularyContains(String word);

    /**
     * Preprocesses the word to embed into an {@link NDArray} to pass into the model.
     *
     * <p>Make sure to call {@link #embedWord(NDArray)} after this.
     *
     * @param manager the manager for the new array
     * @param word the word to embed
     * @return the word that is ready to embed
     */
    NDArray preprocessWordToEmbed(NDManager manager, String word);

    /**
     * Embeds a word.
     *
     * @param manager the manager for the embedding array
     * @param word the word to embed
     * @return the embedded word
     * @throws ai.djl.modality.nlp.embedding.EmbeddingException if there is an error while trying to
     *     embed
     */
    default NDArray embedWord(NDManager manager, String word) throws EmbeddingException {
        return embedWord(preprocessWordToEmbed(manager, word));
    }

    /**
     * Embeds the word after preprocessed using {@link #preprocessWordToEmbed(NDManager, String)}.
     * This method must only be used for pre-trained word embeddings.
     *
     * @param word the word to embed
     * @return the embedded word
     * @throws ai.djl.modality.nlp.embedding.EmbeddingException if there is an error while trying to
     *     embed
     */
    NDArray embedWord(NDArray word) throws EmbeddingException;

    /**
     * Returns the closest matching word for the given index.
     *
     * @param word the word embedding to find the matching string word for.
     * @return a word similar to the passed in embedding
     * @throws EmbeddingException if the input is not an unembeddable index
     */
    String unembedWord(NDArray word) throws EmbeddingException;
}
