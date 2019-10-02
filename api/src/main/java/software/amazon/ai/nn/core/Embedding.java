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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.AbstractBlock;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.util.PairList;

/**
 * An Embedding block map a collection of items to 1-Dimensional representative {@link NDArray}s.
 *
 * @param <T> The type of item that should be embedded and map to the array
 */
public class Embedding<T> extends AbstractBlock {

    private static final byte VERSION = 1;

    private int embeddingSize;
    private boolean useDefault;
    private DataType dataType;
    private Map<T, Integer> embedder;
    private int numItems;

    private Parameter embedding;

    public Embedding(Builder<T> builder) {
        embeddingSize = builder.getEmbeddingSize();
        useDefault = builder.isUseDefault();
        dataType = builder.getDataType();
        embedding = new Parameter("embedding", this, ParameterType.WEIGHT);
        embedder = new ConcurrentHashMap<>(builder.getItems().size());
        numItems = 0;
        if (useDefault) {
            numItems++;
        }
        for (T item : builder.getItems()) {
            embedder.put(item, numItems++);
        }
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return inputs[0].addAll(new Shape(embeddingSize));
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.singletonList(embedding);
    }

    @Override
    public void beforeInitialize(NDList inputs) {}

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        if ("embedding".equals(name)) {
            return new Shape(numItems, embeddingSize);
        }
        throw new IllegalArgumentException("Invalid parameter name");
    }

    /**
     * Finds the embedding of items as a {@link NDArray}.
     *
     * @param manager The manager to create the new NDArray
     * @param items The items to retrieve the embeddings for
     * @return Returns a 3D NDArray where the first two embeddingSize correspond to the items, and
     *     the last dimension is the embedding.
     */
    public NDArray forward(NDManager manager, T[][] items) {
        return forward(new NDList(manager.create(embed(items)))).head();
    }

    /**
     * Finds the embedding of items as a {@link NDArray}.
     *
     * @param manager The manager to create the new NDArray
     * @param items The items to retrieve the embeddings for
     * @return Returns a 2D NDArray where the first dimension corresponds to the items, and the last
     *     dimension is the embedding.
     */
    public NDArray forward(NDManager manager, T[] items) {
        return forward(new NDList(manager.create(embed(items)))).head();
    }

    /**
     * Finds the embedding of an item as a {@link NDArray}.
     *
     * @param manager The manager to create the new NDArray
     * @param item The item to retrieve the embedding for
     * @return Returns the 1D NDArray of the embedding
     */
    public NDArray forward(NDManager manager, T item) {
        return forward(new NDList(manager.create(embed(item)))).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        NDList opInputs = opInputs(inputs);

        NDArrayEx ex = opInputs.head().getNDArrayInternal();
        NDList result = ex.embedding(opInputs, numItems, embeddingSize, dataType, params);
        if (inputs.head().getShape().dimension() == 0) {
            result = new NDList(result.head().reshape(embeddingSize));
        }
        return result;
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        embedding.save(os);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        embedding.load(manager, is);
    }

    private NDList opInputs(NDList inputs) {
        initialize(inputs);
        NDArray items = inputs.get(0);
        if (items.getShape().dimension() == 0) {
            return new NDList(items.reshape(1), embedding.getArray());
        }
        return new NDList(items, embedding.getArray());
    }

    private int[][] embed(T[][] items) {
        return Arrays.stream(items).map(this::embed).toArray(int[][]::new);
    }

    private int[] embed(T[] items) {
        return Arrays.stream(items).mapToInt(this::embed).toArray();
    }

    private int embed(T value) {
        if (embedder.containsKey(value)) {
            return embedder.get(value);
        } else {
            if (useDefault) {
                return 0;
            } else {
                throw new IllegalArgumentException("The provided item was not found");
            }
        }
    }

    /**
     * The Builder to construct a {@link Embedding} type of {@link Block}.
     *
     * @param <T> The type of object to embed
     */
    public static final class Builder<T> {

        private Collection<T> items;
        private int embeddingSize;
        private boolean useDefault = true;
        private DataType dataType = DataType.FLOAT32;

        public Collection<T> getItems() {
            return items;
        }

        public int getEmbeddingSize() {
            return embeddingSize;
        }

        public boolean isUseDefault() {
            return useDefault;
        }

        public DataType getDataType() {
            return dataType;
        }

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
            return new Embedding<>(this);
        }
    }
}
