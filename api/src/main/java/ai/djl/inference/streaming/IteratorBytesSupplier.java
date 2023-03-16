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
package ai.djl.inference.streaming;

import ai.djl.ndarray.BytesSupplier;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

/**
 * An {@link IteratorBytesSupplier} is a streaming {@link BytesSupplier} suitable for synchronous
 * usage.
 */
public class IteratorBytesSupplier implements BytesSupplier, Iterator<byte[]> {

    private Iterator<BytesSupplier> sources;

    /**
     * Constructs an {@link IteratorBytesSupplier}.
     *
     * @param sources the source suppliers
     */
    public IteratorBytesSupplier(Iterator<BytesSupplier> sources) {
        this.sources = sources;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasNext() {
        return sources.hasNext();
    }

    /** {@inheritDoc} */
    @Override
    public byte[] next() {
        return sources.next().getAsBytes();
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(getAsBytes());
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getAsBytes() {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            while (hasNext()) {
                bos.write(next());
            }
            return bos.toByteArray();
        } catch (IOException e) {
            throw new AssertionError("Failed to read BytesSupplier", e);
        }
    }
}
