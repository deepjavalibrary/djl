/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/** Represents a supplier of {@code byte[]}. */
public interface BytesSupplier {

    /**
     * Returns the {@code byte[]} presentation of the object.
     *
     * @return the {@code byte[]} presentation of the object
     */
    default byte[] getAsBytes() {
        ByteBuffer bb = toByteBuffer();
        if (bb.hasArray() && bb.remaining() == bb.capacity()) {
            return bb.array();
        }
        byte[] buf = new byte[bb.remaining()];
        bb.get(buf);
        return buf;
    }

    /**
     * Returns the {@code String} presentation of the object.
     *
     * @return the {@code String} presentation of the object
     */
    default String getAsString() {
        return new String(getAsBytes(), StandardCharsets.UTF_8);
    }

    /**
     * Returns the {@code ByteBuffer} presentation of the object.
     *
     * @return the {@code ByteBuffer} presentation of the object
     */
    ByteBuffer toByteBuffer();

    /**
     * Wraps a byte array into a {code BytesSupplier}.
     *
     * @param buf the byte array that will back this {code BytesSupplier}
     * @return a {@code ByteBuffer}
     */
    static BytesSupplier wrap(byte[] buf) {
        return new BytesSupplierImpl(buf);
    }

    /**
     * Wraps a string into a {code BytesSupplier}.
     *
     * @param value the string that will back this {code BytesSupplier}
     * @return a {@code ByteBuffer}
     */
    static BytesSupplier wrap(String value) {
        return new BytesSupplierImpl(value);
    }
}
