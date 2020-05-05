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
package ai.djl.nn.core;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * An Embedding maps elements of type T to a 1-Dimensional representative {@link NDArray}s.
 *
 * @param <T> the type of item that should be embedded
 */
public interface AbstractEmbedding<T> {

    /**
     * Returns whether an item is in the embedding.
     *
     * @param item the item to test
     * @return true if the item is in the embedding
     */
    boolean hasItem(T item);

    /**
     * Embeds an array of items.
     *
     * @param manager the manager for the new embeddings
     * @param items the items to embed
     * @return the embedding {@link NDArray} of Shape(items.length, embeddingSize)
     */
    NDArray embed(NDManager manager, T[] items);
}
