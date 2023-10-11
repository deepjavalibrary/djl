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

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * A utility for reading a {@link ByteBuffer} with an {@link InputStream}.
 *
 * @see <a href="https://stackoverflow.com/a/6603018">source</a>
 */
public class ByteBufferBackedInputStream extends InputStream {
    ByteBuffer buf;

    /**
     * Constructs a new {@link ByteBufferBackedInputStream}.
     *
     * @param buf the backing buffer
     */
    public ByteBufferBackedInputStream(ByteBuffer buf) {
        this.buf = buf;
    }

    /** {@inheritDoc} */
    @Override
    public int read() {
        if (!buf.hasRemaining()) {
            return -1;
        }
        return buf.get() & 0xFF;
    }

    /** {@inheritDoc} */
    @Override
    public int read(byte[] bytes, int off, int len) {
        if (!buf.hasRemaining()) {
            return -1;
        }

        len = Math.min(len, buf.remaining());
        buf.get(bytes, off, len);
        return len;
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void mark(int readlimit) {
        buf.mark();
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void reset() throws IOException {
        buf.reset();
    }

    /** {@inheritDoc} */
    @Override
    public boolean markSupported() {
        return true;
    }
}
