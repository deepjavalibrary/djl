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
package ai.djl.nn.core;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.nn.convolutional.Conv2D.Builder;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An Embedding block map a collection of items to 1-Dimensional representative {@link NDArray}s.
 *
 * @param <T> the type of item that should be embedded and map to the array
 */
public class Embedding<T> extends ParameterBlock {

    private static final byte VERSION = 2;

    private int embeddingSize;
    private boolean useDefault;
    private DataType dataType;
    private Map<T, Integer> embedder;
    private int numItems;

    private Parameter embedding;

    Embedding(Builder<T> builder) {
        embeddingSize = builder.embeddingSize;
        useDefault = builder.useDefault;
        dataType = builder.dataType;
        embedding = new Parameter("embedding", this, ParameterType.WEIGHT);
        embedder = new ConcurrentHashMap<>(builder.items.size());
        numItems = 0;
        if (useDefault) {
            numItems++;
        }
        for (T item : builder.items) {
            embedder.put(item, numItems++);
        }
        inputShapes = new Shape[] {new Shape(-1)};
    }

    /**
     * Constructs a pretrained embedding.
     *
     * @param embedding the embedding array
     * @param items the items in the embedding (in matching order to the embedding array)
     */
    public Embedding(NDArray embedding, List<T> items) {
        embeddingSize = Math.toIntExact(embedding.getShape().get(1));
        useDefault = false;
        dataType = embedding.getDataType();
        this.embedding = new Parameter("embedding", this, ParameterType.WEIGHT);
        this.embedding.setArray(embedding);
        numItems = items.size();
        embedder = new ConcurrentHashMap<>(numItems);
        for (int i = 0; i < items.size(); i++) {
            embedder.put(items.get(i), i);
        }
        inputShapes = new Shape[] {new Shape(-1)};
    }

    /**
     * Creates a builder to build an {@link Embedding}.
     *
     * @return a new builder
     */
    public static Embedding.Builder<?> builder() {
        return new Embedding.Builder<>();
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[] {inputShapes[0].addAll(new Shape(embeddingSize))};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.singletonList(embedding);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        if ("embedding".equals(name)) {
            return new Shape(numItems, embeddingSize);
        }
        throw new IllegalArgumentException("Invalid parameter name");
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        NDList opInputs = opInputs(parameterStore, inputs);

        NDArrayEx ex = opInputs.head().getNDArrayInternal();
        NDList result = ex.embedding(opInputs, numItems, embeddingSize, dataType, params);
        if (inputs.singletonOrThrow().getShape().dimension() == 0) {
            result = new NDList(result.singletonOrThrow().reshape(embeddingSize));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        saveInputShapes(os);
        embedding.save(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version == VERSION) {
            readInputShapes(is);
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        embedding.load(manager, is);
    }

    /**
     * Returns whether an item is in the embedding.
     *
     * @param item the item to test
     * @return true if the item is in the embedding
     */
    public boolean hasItem(T item) {
        return embedder.containsKey(item);
    }

    private NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        NDArray items = inputs.singletonOrThrow();
        Device device = items.getDevice();

        NDList ret = new NDList(2);
        if (items.getShape().dimension() == 0) {
            ret.add(items.reshape(1));
        } else {
            ret.add(items);
        }
        ret.add(parameterStore.getValue(embedding, device));
        return ret;
    }

    /**
     * Embeds an array of items.
     *
     * @param manager the manager for the new embeddings
     * @param items the items to embed
     * @return the embedding {@link NDArray} of Shape(items.length)
     */
    public NDArray embed(NDManager manager, T[] items) {
        return manager.create(Arrays.stream(items).mapToInt(this::embedHelper).toArray());
    }

    /**
     * Embeds an item.
     *
     * @param manager the manager for the new embedding
     * @param item the item to embed
     * @return the embedding {@link NDArray} of Shape()
     */
    public NDArray embed(NDManager manager, T item) {
        return manager.create(embedHelper(item));
    }

    private int embedHelper(T value) {
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
     * @param <T> the type of object to embed
     */
    public static final class Builder<T> {

        private Class<T> embeddingType;
        private Collection<T> items;
        private int embeddingSize;
        private boolean useDefault = true;
        private DataType dataType = DataType.FLOAT32;

        Builder() {}

        private Builder(Class<T> embeddingType, Builder<?> parent) {
            this.embeddingType = embeddingType;
            this.embeddingSize = parent.embeddingSize;
            this.useDefault = parent.useDefault;
            this.dataType = parent.dataType;
        }

        /**
         * Returns the embedded type.
         *
         * @return the embedded type
         */
        public Class<T> getEmbeddingType() {
            return embeddingType;
        }

        /**
         * Creates a new {@link Builder} with the specified embedding type.
         *
         * @param embeddingType the embedding class
         * @param <T> the embedding type
         * @return a new {@link Builder} class with the specified embedding type
         */
        public <T> Builder<T> setType(Class<T> embeddingType) {
            return new Builder<>(embeddingType, this);
        }

        /**
         * Sets the collection of items that should feature embeddings.
         *
         * @param items a collection containing all the items that embeddings should be created for
         * @return this Builder
         */
        public Builder<T> setItems(Collection<T> items) {
            this.items = items;
            return this;
        }

        /**
         * Sets the size of the embeddings.
         *
         * @param embeddingSize the size of the 1D embedding array
         * @return this Builder
         */
        public Builder<T> setEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        /**
         * Sets whether to use a default embedding for undefined items (default true).
         *
         * @param useDefault true to provide a default embedding and false to throw an {@link
         *     IllegalArgumentException} when the item can not be found
         * @return this Builder
         */
        public Builder<T> optUseDefault(boolean useDefault) {
            this.useDefault = useDefault;
            return this;
        }

        /**
         * Sets the data type of the embedding arrays (default is Float32).
         *
         * @param dataType the dataType to use for the embedding
         * @return this Builder
         */
        public Builder<T> optDataType(DataType dataType) {
            this.dataType = dataType;
            return this;
        }

        /**
         * Builds the {@link Embedding}.
         *
         * @return the constructed {@code Embedding}
         * @throws IllegalArgumentException if all required parameters (items, embeddingSize) have
         *     not been set
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
