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

import ai.djl.util.PairList;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A class stores the generic input data for inference. */
public class Input {

    private String requestId;
    private Map<String, String> properties;
    private PairList<String, byte[]> content;

    /**
     * Constructs a {@code Input} with specified {@code requestId}.
     *
     * @param requestId the requestId of the input
     */
    public Input(String requestId) {
        this.requestId = requestId;
        properties = new ConcurrentHashMap<>();
        content = new PairList<>();
    }

    /**
     * Returns the requestId of the input.
     *
     * @return the requestId of the input
     */
    public String getRequestId() {
        return requestId;
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
        if (properties == null) {
            properties = new ConcurrentHashMap<>();
        }
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
        return properties.getOrDefault(key.toLowerCase(Locale.ROOT), defaultValue);
    }

    /**
     * Returns the content of the input.
     *
     * <p>A {@code Input} may contains multiple data.
     *
     * @return the content of the input
     */
    public PairList<String, byte[]> getContent() {
        return content;
    }

    /**
     * Sets the content of the input.
     *
     * @param content the content of the input
     */
    public void setContent(PairList<String, byte[]> content) {
        this.content = content;
    }

    /**
     * Appends an item at the end of the input.
     *
     * @param data data to be added
     */
    public void addData(byte[] data) {
        addData(null, data);
    }

    /**
     * Adds a key/value pair to the input content.
     *
     * @param key key with which the specified data is to be added
     * @param data data to be added with the specified key
     */
    public void addData(String key, byte[] data) {
        if (content == null) {
            content = new PairList<>();
        }
        content.add(key, data);
    }

    /**
     * Inserts the specified element at the specified position in the input.
     *
     * @param index the index at which the specified element is to be inserted
     * @param data data to be added with the specified key
     */
    public void addData(int index, byte[] data) {
        if (content == null) {
            content = new PairList<>();
        }
        content.add(index, null, data);
    }
}
