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
package ai.djl.basicdataset.utils;

import java.nio.FloatBuffer;

/** A float buffer that can dynamically change it's capacity. */
public class DynamicBuffer {

    private FloatBuffer buffer;
    private int length;

    /** Constructs a new instance of {@code DynamicBuffer}. */
    public DynamicBuffer() {
        buffer = FloatBuffer.allocate(128);
    }

    /**
     * Writes the given float into this buffer at the current position.
     *
     * @param f the float to be written
     * @return this buffer
     */
    public DynamicBuffer put(float f) {
        ++length;
        if (buffer.capacity() == length) {
            FloatBuffer buf = buffer;
            buf.rewind();
            buffer = FloatBuffer.allocate(length * 2);
            buffer.put(buf);
        }
        buffer.put(f);
        return this;
    }

    /**
     * Returns a {@code FloatBuffer} that contains all the data.
     *
     * @return a {@code FloatBuffer}
     */
    public FloatBuffer getBuffer() {
        buffer.rewind();
        buffer.limit(length);
        return buffer;
    }

    /**
     * Returns the buffer size.
     *
     * @return the buffer size
     */
    public int getLength() {
        return length;
    }
}
