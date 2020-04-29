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
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An Embedding block map a collection of items to 1-Dimensional representative {@link NDArray}s.
 *
 * @param <T> the type of item that should be embedded and map to the array
 */
public abstract class Embedding<T> extends ParameterBlock {

    private static final byte VERSION = 3;

    protected int embeddingSize;
    protected boolean useDefault;
    protected boolean sparseGrad;
    protected DataType dataType;
    protected Map<T, Integer> embedder;
    protected Map<Integer, T> unembedder;
    protected int numItems;

    protected Parameter embedding;

    protected Embedding(BaseBuilder<T, ?> baseBuilder) {
        embeddingSize = baseBuilder.embeddingSize;
        useDefault = baseBuilder.useDefault;
        sparseGrad = baseBuilder.sparseGrad;
        dataType = baseBuilder.dataType;
        embedding =
                new Parameter(
                        "embedding",
                        this,
                        ParameterType.WEIGHT,
                        true,
                        sparseGrad ? SparseFormat.ROW_SPARSE : SparseFormat.DENSE);
        embedder = new ConcurrentHashMap<>();
        unembedder = new ConcurrentHashMap<>();
        if (useDefault) {
            numItems++;
        }
        for (T item : baseBuilder.items) {
            embedder.put(item, numItems);
            unembedder.put(numItems++, item);
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
        this(embedding, items, true);
    }

    /**
     * Constructs a pretrained embedding.
     *
     * @param embedding the embedding array
     * @param items the items in the embedding (in matching order to the embedding array)
     * @param sparseGrad whether to compute row sparse gradient in the backward calculation
     */
    public Embedding(NDArray embedding, List<T> items, boolean sparseGrad) {
        embeddingSize = Math.toIntExact(embedding.getShape().get(1));
        useDefault = false;
        this.sparseGrad = sparseGrad;
        dataType = embedding.getDataType();
        this.embedding =
                new Parameter(
                        "embedding",
                        this,
                        ParameterType.WEIGHT,
                        true,
                        sparseGrad ? SparseFormat.ROW_SPARSE : SparseFormat.DENSE);
        this.embedding.setArray(embedding);
        numItems = items.size();
        embedder = new ConcurrentHashMap<>(numItems);
        unembedder = new ConcurrentHashMap<>(numItems);
        for (int i = 1; i <= items.size(); i++) {
            embedder.put(items.get(i), i);
            unembedder.put(i, items.get(i));
        }
        inputShapes = new Shape[] {new Shape(-1)};
    }

    /**
     * Encodes an object of input type into a byte array. This is used in saving and loading the
     * {@link Embedding} objects.
     *
     * @param input the input object to be encoded
     * @return the encoded byte array.
     * @throws IOException if there is an error while encoding
     */
    public abstract byte[] encode(T input) throws IOException;

    /**
     * Decodes the given byte array into an object of input parameter type.
     *
     * @param byteArray the byte array to be decoded
     * @return the decode object of input parameter type
     * @throws IOException if there was an error while decoding
     */
    public abstract T decode(byte[] byteArray) throws IOException;

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
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList opInputs = opInputs(parameterStore, inputs);

        NDArrayEx ex = opInputs.head().getNDArrayInternal();
        NDList result =
                ex.embedding(opInputs, numItems, embeddingSize, sparseGrad, dataType, params);
        if (inputs.head().getShape().dimension() == 0) {
            result = new NDList(result.singletonOrThrow().reshape(embeddingSize));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        saveInputShapes(os);
        os.writeBoolean(useDefault);
        os.writeBoolean(sparseGrad);
        os.writeUTF(dataType.toString());
        Set<Map.Entry<T, Integer>> embedderEntrySet = embedder.entrySet();
        os.writeInt(embedderEntrySet.size());
        for (Map.Entry<T, Integer> entry : embedderEntrySet) {
            byte[] encodedKey = encode(entry.getKey());
            os.writeInt(encodedKey.length);
            os.write(encodedKey);
            os.writeInt(embedder.get(entry.getKey()));
        }
        embedding.save(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version == VERSION) {
            readInputShapes(is);
            useDefault = is.readBoolean();
            sparseGrad = is.readBoolean();
            dataType = DataType.valueOf(is.readUTF().toUpperCase(Locale.ENGLISH));
            embedder = new ConcurrentHashMap<>();
            unembedder = new ConcurrentHashMap<>();
            int embedderSize = is.readInt();
            for (int i = 0; i < embedderSize; i++) {
                int encodedKeySize = is.readInt();
                byte[] encodedKey = new byte[encodedKeySize];
                if (is.read(encodedKey) != encodedKey.length) {
                    throw new MalformedModelException("Model data is malformed");
                }
                int value = is.readInt();
                embedder.put(decode(encodedKey), value);
                unembedder.put(value, decode(encodedKey));
            }
        } else if (version == 2) {
            readInputShapes(is);
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        embedding.load(manager, is);
        numItems = (int) embedding.getArray().getShape().get(0);
        embeddingSize = (int) embedding.getArray().getShape().get(1);
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
        NDArray items = inputs.head();
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
     * @param item the item to embed
     * @return the embedding {@link NDArray} of Shape()
     */
    public int embed(T item) {
        return embedHelper(item);
    }

    /**
     * Returns the item corresponding to the given index.
     *
     * @param index the index
     * @return the item corresponding to the given index
     */
    public Optional<T> unembed(int index) {
        return Optional.ofNullable(unembedder.get(index));
    }

    private Integer embedHelper(T value) {
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
    public abstract static class BaseBuilder<T, B extends BaseBuilder<T, B>> {

        protected Class<T> embeddingType;
        protected List<T> items = new ArrayList<>();
        protected int embeddingSize;
        protected boolean useDefault = true;
        protected boolean sparseGrad = true;
        protected DataType dataType = DataType.FLOAT32;

        protected BaseBuilder() {}

        /**
         * Returns the embedded type.
         *
         * @return the embedded type
         */
        public Class<T> getEmbeddingType() {
            return embeddingType;
        }

        /**
         * Creates a new {@link BaseBuilder} with the specified embedding type.
         *
         * @param embeddingType the embedding class
         * @return a new {@link BaseBuilder} class with the specified embedding type
         */
        protected abstract B setType(Class<T> embeddingType);

        /**
         * Sets the collection of items that should feature embeddings.
         *
         * @param items a {@link List} containing all the items that embeddings should be created
         *     for
         * @return this Builder
         */
        public B setItems(List<T> items) {
            this.items = items;
            return self();
        }

        /**
         * Sets the size of the embeddings.
         *
         * @param embeddingSize the size of the 1D embedding array
         * @return this Builder
         */
        public B setEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return self();
        }

        /**
         * Sets whether to use a default embedding for undefined items (default true).
         *
         * @param useDefault true to provide a default embedding and false to throw an {@link
         *     IllegalArgumentException} when the item can not be found
         * @return this Builder
         */
        public B optUseDefault(boolean useDefault) {
            this.useDefault = useDefault;
            return self();
        }

        /**
         * Sets the optional parameter whether to compute row sparse gradient in the backward
         * calculation. If set to True, the grad’s storage type is row_sparse.
         *
         * @param sparseGrad whether to compute row sparse gradient in the backward calculation
         * @return this Builder
         */
        public B optSparseGrad(boolean sparseGrad) {
            this.sparseGrad = sparseGrad;
            return self();
        }

        /**
         * Sets the data type of the embedding arrays (default is Float32).
         *
         * @param dataType the dataType to use for the embedding
         * @return this Builder
         */
        public B optDataType(DataType dataType) {
            this.dataType = dataType;
            return self();
        }

        /**
         * Returns this {code Builder} object.
         *
         * @return this {@code BaseBuilder}
         */
        protected abstract B self();
    }
}
