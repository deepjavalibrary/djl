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
package ai.djl.modality;

import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.PairList;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.TreeMap;

/** A class stores the generic input data for inference. */
public class Input {

    private Map<String, String> properties;
    private PairList<String, BytesSupplier> content;

    /** Constructs a new {@code Input} instance. */
    public Input() {
        properties = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        content = new PairList<>();
    }

    /**
     * Returns the properties of the input.
     *
     * @return the properties of the input
     */
    public Map<String, String> getProperties() {
        return properties;
    }

    /**
     * Sets the properties of the input.
     *
     * @param properties the properties of the input
     */
    public void setProperties(Map<String, String> properties) {
        this.properties = properties;
    }

    /**
     * Adds a property to the input.
     *
     * @param key key with which the specified value is to be added
     * @param value value to be added with the specified key
     */
    public void addProperty(String key, String value) {
        properties.put(key, value);
    }

    /**
     * Returns the value to which the specified key is mapped.
     *
     * @param key the key whose associated value is to be returned
     * @param defaultValue the default mapping of the key
     * @return the value to which the specified key is mapped
     */
    public String getProperty(String key, String defaultValue) {
        return properties.getOrDefault(key, defaultValue);
    }

    /**
     * Returns the content of the input.
     *
     * <p>A {@code Input} may contains multiple data.
     *
     * @return the content of the input
     */
    public PairList<String, BytesSupplier> getContent() {
        return content;
    }

    /**
     * Sets the content of the input.
     *
     * @param content the content of the input
     */
    public void setContent(PairList<String, BytesSupplier> content) {
        this.content = content;
    }

    /**
     * Appends an item at the end of the input.
     *
     * @param data data to be added
     */
    public void add(byte[] data) {
        add(BytesSupplier.wrap(data));
    }

    /**
     * Appends an item at the end of the input.
     *
     * @param data data to be added
     */
    public void add(String data) {
        add(BytesSupplier.wrap(data.getBytes(StandardCharsets.UTF_8)));
    }

    /**
     * Appends an item at the end of the input.
     *
     * @param data data to be added
     */
    public void add(BytesSupplier data) {
        add(null, data);
    }

    /**
     * Adds a key/value pair to the input content.
     *
     * @param key key with which the specified data is to be added
     * @param data data to be added with the specified key
     */
    public void add(String key, byte[] data) {
        add(key, BytesSupplier.wrap(data));
    }

    /**
     * Adds a key/value pair to the input content.
     *
     * @param key key with which the specified data is to be added
     * @param data data to be added with the specified key
     */
    public void add(String key, String data) {
        add(key, BytesSupplier.wrap(data));
    }

    /**
     * Adds a key/value pair to the input content.
     *
     * @param key key with which the specified data is to be added
     * @param data data to be added with the specified key
     */
    public void add(String key, BytesSupplier data) {
        content.add(key, data);
    }

    /**
     * Inserts the specified element at the specified position in the input.
     *
     * @param index the index at which the specified element is to be inserted
     * @param key key with which the specified data is to be added
     * @param data data to be added with the specified key
     */
    public void add(int index, String key, BytesSupplier data) {
        content.add(index, key, data);
    }

    /**
     * Returns the default data item.
     *
     * @return the default data item
     */
    public BytesSupplier getData() {
        if (content.isEmpty()) {
            return null;
        }

        BytesSupplier data = get("data");
        if (data == null) {
            return get(0);
        }
        return data;
    }

    /**
     * Returns the default data as {@code NDList}.
     *
     * @param manager {@link NDManager} used to create this {@code NDArray}
     * @return the default data as {@code NDList}
     */
    public NDList getDataAsNDList(NDManager manager) {
        if (content.isEmpty()) {
            return null;
        }

        int index = content.indexOf("data");
        if (index < 0) {
            index = 0;
        }
        return getAsNDList(manager, index);
    }

    /**
     * Returns the element for the first key found in the {@code Input}.
     *
     * @param key the key of the element to get
     * @return the element for the first key found in the {@code Input}
     */
    public BytesSupplier get(String key) {
        return content.get(key);
    }

    /**
     * Returns the element at the specified position in the {@code Input}.
     *
     * @param index the index of the element to return
     * @return the element at the specified position in the {@code Input}
     */
    public BytesSupplier get(int index) {
        return content.valueAt(index);
    }

    /**
     * Returns the value as {@code byte[]} for the first key found in the {@code Input}.
     *
     * @param key the key of the element to get
     * @return the value as {@code byte[]} for the first key found in the {@code Input}
     */
    public byte[] getAsBytes(String key) {
        BytesSupplier data = content.get(key);
        if (data == null) {
            return null;
        }
        return data.getAsBytes();
    }

    /**
     * Returns the value as {@code byte[]} at the specified position in the {@code Input}.
     *
     * @param index the index of the element to return
     * @return the value as {@code byte[]} at the specified position in the {@code Input}
     */
    public byte[] getAsBytes(int index) {
        return content.valueAt(index).getAsBytes();
    }

    /**
     * Returns the value as {@code byte[]} for the first key found in the {@code Input}.
     *
     * @param key the key of the element to get
     * @return the value as {@code byte[]} for the first key found in the {@code Input}
     */
    public String getAsString(String key) {
        BytesSupplier data = content.get(key);
        if (data == null) {
            return null;
        }
        return data.getAsString();
    }

    /**
     * Returns the value as {@code byte[]} at the specified position in the {@code Input}.
     *
     * @param index the index of the element to return
     * @return the value as {@code byte[]} at the specified position in the {@code Input}
     */
    public String getAsString(int index) {
        return content.valueAt(index).getAsString();
    }

    /**
     * Returns the value as {@code NDArray} for the first key found in the {@code Input}.
     *
     * @param manager {@link NDManager} used to create this {@code NDArray}
     * @param key the key of the element to get
     * @return the value as {@code NDArray} for the first key found in the {@code Input}
     */
    public NDArray getAsNDArray(NDManager manager, String key) {
        int index = content.indexOf(key);
        if (index < 0) {
            return null;
        }
        return getAsNDArray(manager, index);
    }

    /**
     * Returns the value as {@code NDArray} at the specified position in the {@code Input}.
     *
     * @param manager {@link NDManager} used to create this {@code NDArray}
     * @param index the index of the element to return
     * @return the value as {@code NDArray} at the specified position in the {@code Input}
     */
    public NDArray getAsNDArray(NDManager manager, int index) {
        BytesSupplier data = content.valueAt(index);
        if (data instanceof NDArray) {
            return (NDArray) data;
        }
        return NDArray.decode(manager, data.getAsBytes());
    }

    /**
     * Returns the value as {@code NDList} for the first key found in the {@code Input}.
     *
     * @param manager {@link NDManager} used to create this {@code NDArray}
     * @param key the key of the element to get
     * @return the value as {@code NDList} for the first key found in the {@code Input}
     */
    public NDList getAsNDList(NDManager manager, String key) {
        int index = content.indexOf(key);
        if (index < 0) {
            return null;
        }
        return getAsNDList(manager, index);
    }

    /**
     * Returns the value as {@code NDList} at the specified position in the {@code Input}.
     *
     * @param manager {@link NDManager} used to create this {@code NDArray}
     * @param index the index of the element to return
     * @return the value as {@code NDList} at the specified position in the {@code Input}
     */
    public NDList getAsNDList(NDManager manager, int index) {
        BytesSupplier data = content.valueAt(index);
        if (data instanceof NDList) {
            return (NDList) data;
        } else if (data instanceof NDArray) {
            return new NDList((NDArray) data);
        }
        return NDList.decode(manager, data.getAsBytes());
    }
}
