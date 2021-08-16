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

import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A class stores the generic inference results. */
public class Output {

    private String requestId;
    private int code;
    private String message;
    private Map<String, String> properties;
    private byte[] content;

    /**
     * Constructs a {@code Output} with specified {@code requestId}, {@code code} and {@code
     * message}.
     *
     * @param code the status code of the output
     * @param message the status message of the output
     */
    public Output(int code, String message) {
        this.code = code;
        this.message = message;
    }

    /**
     * Sets the requestId of the output.
     *
     * @param requestId the requestId of the output
     */
    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    /**
     * Returns the requestId of the output.
     *
     * @return the requestId of the output
     */
    public String getRequestId() {
        return requestId;
    }

    /**
     * Returns the status code of the output.
     *
     * @return the status code of the output
     */
    public int getCode() {
        return code;
    }

    /**
     * Sets the status code of the output.
     *
     * @param code the status code of the output
     */
    public void setCode(int code) {
        this.code = code;
    }

    /**
     * Returns the status code of the output.
     *
     * @return the status code of the output
     */
    public String getMessage() {
        return message;
    }

    /**
     * Sets the status message of the output.
     *
     * @param message the status message of the output
     */
    public void setMessage(String message) {
        this.message = message;
    }

    /**
     * Returns the properties of the output.
     *
     * @return the properties of the output
     */
    public Map<String, String> getProperties() {
        if (properties == null) {
            return Collections.emptyMap();
        }
        return properties;
    }

    /**
     * Sets the properties of the output.
     *
     * @param properties the properties of the output
     */
    public void setProperties(Map<String, String> properties) {
        this.properties = properties;
    }

    /**
     * Adds a property to the output.
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
     * Returns the content of the input.
     *
     * @return the content of the input
     */
    public byte[] getContent() {
        return content;
    }

    /**
     * Sets the content of the input.
     *
     * @param content the content of the input
     */
    public void setContent(byte[] content) {
        this.content = content;
    }

    /**
     * Sets the content of the input with string value.
     *
     * @param content the content of the input in string
     */
    public void setContent(String content) {
        this.content = content.getBytes(StandardCharsets.UTF_8);
    }
}
