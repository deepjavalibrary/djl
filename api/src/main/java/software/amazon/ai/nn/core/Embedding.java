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
package software.amazon.ai.nn.core;

import java.util.Collection;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.nn.Block;

/**
 * An Embedding block map a collection of items to 1-Dimensional representative {@link NDArray}s.
 *
 * @param <T> The type of item that should be embedded and map to the array
 */
public interface Embedding<T> extends Block {

    /**
     * Finds the embedding of items as a {@link NDArray}.
     *
     * @param manager The manager to create the new NDArray
     * @param items The items to retrieve the embeddings for
     * @return Returns a 3D NDArray where the first two embeddingSize correspond to the items, and
     *     the last dimension is the embedding.
     */
    NDArray forward(NDManager manager, T[][] items);

    /**
     * Finds the embedding of items as a {@link NDArray}.
     *
     * @param manager The manager to create the new NDArray
     * @param items The items to retrieve the embeddings for
     * @return Returns a 2D NDArray where the first dimension corresponds to the items, and the last
     *     dimension is the embedding.
     */
    NDArray forward(NDManager manager, T[] items);

    /**
     * Finds the embedding of an item as a {@link NDArray}.
     *
     * @param manager The manager to create the new NDArray
     * @param item The item to retrieve the embedding for
     * @return Returns the 1D NDArray of the embedding
     */
    NDArray forward(NDManager manager, T item);

    /**
     * The Builder to construct a {@link Embedding} type of {@link Block}.
     *
     * @param <T> The type of object to embed
     */
    class Builder<T> {

        Collection<T> items;
        int embeddingSize;
        boolean useDefault = true;
        DataType dataType = DataType.FLOAT32;

        /**
         * Sets the collection of items that should feature embeddings.
         *
         * @param items A collection containing all the items that embedddings should be created
         *     for.
         * @return Returns this Builder
         */
        public Builder<T> setItems(Collection<T> items) {
            this.items = items;
            return this;
        }

        /**
         * Sets the size of the embeddings.
         *
         * @param embeddingSize The size of the 1D embedding array
         * @return Returns this Builder
         */
        public Builder<T> setEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        /**
         * Sets whether to use a default embedding for undefined items (default true).
         *
         * @param useDefault True to provide a default embedding and false to throw an {@link
         *     IllegalArgumentException} when the item can not be found
         * @return Returns this Builder
         */
        public Builder<T> setUseDefault(boolean useDefault) {
            this.useDefault = useDefault;
            return this;
        }

        /**
         * Sets the data type of the embedding arrays (default is Float32).
         *
         * @param dataType The dataType to use for the embedding
         * @return Returns this Builder
         */
        public Builder<T> setDataType(DataType dataType) {
            this.dataType = dataType;
            return this;
        }

        /**
         * Builds the {@link Embedding}.
         *
         * @return Returns the constructed {@code Embedding}
         * @throws IllegalArgumentException Thrown if all required parameters (items, embeddingSize)
         *     have not been set
         */
        public Embedding<T> build() {
            if (items == null) {
                throw new IllegalArgumentException("You must specify the items to embed");
            }
            if (embeddingSize == 0) {
                throw new IllegalArgumentException("You must specify the embedding size");
            }
            return Engine.getInstance()
                    .getNNIndex()
                    .embedding(items, embeddingSize, useDefault, dataType);
        }
    }
}
