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
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** A {link BytesSupplier} that supports chunked reading. */
public class ChunkedBytesSupplier implements BytesSupplier {

    private LinkedBlockingQueue<BytesSupplier> queue;
    private AtomicBoolean completed;

    /** Constructs a new {code ChunkedBytesSupplier} instance. */
    public ChunkedBytesSupplier() {
        queue = new LinkedBlockingQueue<>();
        completed = new AtomicBoolean();
    }

    /**
     * Appends content to the {@code BytesSupplier}.
     *
     * @param data bytes to append
     * @param lastChunk true if this is the last chunk
     */
    public void appendContent(byte[] data, boolean lastChunk) {
        queue.offer(BytesSupplier.wrap(data));
        if (lastChunk) {
            completed.set(true);
        }
    }

    /**
     * Returns {@code true} if has more chunk.
     *
     * @return {@code true} if has more chunk
     */
    public boolean hasNext() {
        return !completed.get() || !queue.isEmpty();
    }

    /**
     * Returns the next chunk.
     *
     * @param timeout the maximum time to wait
     * @param unit the time unit of the timeout argument
     * @return the next chunk
     * @throws InterruptedException if the thread is interrupted
     */
    public byte[] nextChunk(long timeout, TimeUnit unit) throws InterruptedException {
        BytesSupplier data = queue.poll(timeout, unit);
        if (data == null) {
            throw new IllegalStateException("Read chunk timeout.");
        }
        return data.getAsBytes();
    }

    /**
     * Retrieves and removes the head of chunk or returns {@code null} if data is not available.
     *
     * @return the head of chunk or returns {@code null} if data is not available
     */
    public byte[] poll() {
        BytesSupplier data = queue.poll();
        return data == null ? null : data.getAsBytes();
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getAsBytes() {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            while (hasNext()) {
                bos.write(nextChunk(1, TimeUnit.MINUTES));
            }
            return bos.toByteArray();
        } catch (IOException | InterruptedException e) {
            throw new AssertionError("Failed to read BytesSupplier", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(getAsBytes());
    }
}
