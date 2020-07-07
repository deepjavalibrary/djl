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

import java.io.IOException;
import java.util.Optional;

/**
 * An {@link AbstractEmbedding} where each embedded item can be assigned an integer index.
 *
 * @param <T> the type of the item that should be embedded
 */
public interface AbstractIndexedEmbedding<T> extends AbstractEmbedding<T> {

    /**
     * Encodes an object of input type into a byte array. This is used in saving and loading the
     * {@link Embedding} objects.
     *
     * @param input the input object to be encoded
     * @return the encoded byte array.
     * @throws IOException if there is an error while encoding
     */
    byte[] encode(T input) throws IOException;

    /**
     * Decodes the given byte array into an object of input parameter type.
     *
     * @param byteArray the byte array to be decoded
     * @return the decode object of input parameter type
     * @throws IOException if there was an error while decoding
     */
    T decode(byte[] byteArray) throws IOException;

    /**
     * Embeds an item.
     *
     * @param item the item to embed
     * @return the index of the item in the embedding
     */
    long embed(T item);

    /**
     * Returns the item corresponding to the given index.
     *
     * @param index the index
     * @return the item corresponding to the given index
     */
    Optional<T> unembed(long index);
}
