/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util;

/** A class containing the string key-value pair. */
public class StringPair extends Pair<String, String> {

    /**
     * Constructs a {@code Pair} instance with key and value.
     *
     * @param key the key
     * @param value the value
     */
    public StringPair(String key, String value) {
        super(key, value);
    }
}
