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
package ai.djl.modality;

import ai.djl.ndarray.BytesSupplier;
import ai.djl.util.Utils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/** A {link BytesSupplier} that supports chunked reading. */
public class ChunkedBytesSupplier implements BytesSupplier {

    private static final long CHUNK_READ_TIMEOUT = getTimeout(); // default 2 minutes
    private static final BytesSupplier LAST_CHUNK = BytesSupplier.wrapAsJson(null);

    private LinkedBlockingQueue<BytesSupplier> queue;

    /** Constructs a new {code ChunkedBytesSupplier} instance. */
    public ChunkedBytesSupplier() {
        queue = new LinkedBlockingQueue<>();
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
            queue.offer(LAST_CHUNK);
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean isChunked() {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getChunkedInput() {
        return new InputStream() {

            private byte[] current;
            private int pos;
            private boolean closed;

            /** {@inheritDoc} */
            @Override
            public int read(byte[] b, int off, int len) throws IOException {
                if (closed) {
                    return -1;
                }

                try {
                    if (current == null || pos >= current.length) {
                        BytesSupplier data = queue.poll(CHUNK_READ_TIMEOUT, TimeUnit.MILLISECONDS);
                        if (data == null) {
                            throw new IOException("Read chunk timeout.");
                        }
                        if (data == LAST_CHUNK) {
                            closed = true;
                            return -1;
                        }
                        current = data.getAsBytes();
                        pos = 0;
                    }

                    int size = Math.min(len, current.length - pos);
                    System.arraycopy(current, pos, b, off, size);
                    pos += size;
                    return size;
                } catch (InterruptedException e) {
                    throw new IOException("Read interrupted", e);
                }
            }

            /** {@inheritDoc} */
            @Override
            public int read() throws IOException {
                if (closed) {
                    return -1;
                }

                if (current != null && pos < current.length) {
                    return current[pos++] & 0xff;
                }
                while (true) {
                    try {
                        BytesSupplier data = queue.poll(CHUNK_READ_TIMEOUT, TimeUnit.MILLISECONDS);
                        if (data == null) {
                            throw new IOException("Read chunk timeout.");
                        }
                        if (data == LAST_CHUNK) {
                            closed = true;
                            return -1;
                        }
                        current = data.getAsBytes();
                        if (current.length == 0) {
                            continue;
                        }
                        pos = 1;
                        return current[0] & 0xff;
                    } catch (InterruptedException e) {
                        throw new IOException("Read timeout", e);
                    }
                }
            }
        };
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getAsBytes() {
        try {
            return Utils.toByteArray(getChunkedInput());
        } catch (IOException e) {
            throw new AssertionError("Failed to read BytesSupplier", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(getAsBytes());
    }

    private static long getTimeout() {
        String timeOut = Utils.getEnvOrSystemProperty("CHUNK_READ_TIMEOUT");
        if (timeOut == null) {
            return Duration.ofSeconds(120).toMillis();
        }
        return Duration.ofSeconds(Integer.parseInt(timeOut)).toMillis();
    }
}
