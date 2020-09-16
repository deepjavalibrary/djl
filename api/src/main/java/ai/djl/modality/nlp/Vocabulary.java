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
package ai.djl.modality.nlp;

/**
 * {@code Vocabulary} is a collection of tokens. The primary purpose of a vocabulary is the map a
 * token to an index.
 */
public interface Vocabulary {

    /**
     * Returns the token corresponding to the given index.
     *
     * @param index the index
     * @return the token corresponding to the given index
     */
    String getToken(long index);

    /**
     * Check if the vocabulary contains a token.
     *
     * @param token String token to be checked
     * @return whether this vocabulary contains the token
     */
    boolean contains(String token);

    /**
     * Returns the index of the given token.
     *
     * @param token the token
     * @return the index of the given token.
     */
    long getIndex(String token);

    /**
     * Returns the size of the {@link Vocabulary}.
     *
     * @return the size of the {@link Vocabulary}
     */
    long size();
}
