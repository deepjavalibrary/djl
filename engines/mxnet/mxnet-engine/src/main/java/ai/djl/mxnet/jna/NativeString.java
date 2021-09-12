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
package ai.djl.mxnet.jna;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import java.nio.charset.Charset;

/**
 * Provides a temporary allocation of an immutable C string (<code>const char*</code> or <code>
 * const wchar_t*</code>) for use when converting a Java String into a native memory function
 * argument.
 */
final class NativeString {

    private static final ObjectPool<NativeString> POOL = new ObjectPool<>(null, null);

    private Memory pointer;

    /**
     * Create a native string (NUL-terminated array of <code>char</code>), using the requested
     * encoding.
     *
     * @param data the bytes of the string
     */
    private NativeString(byte[] data) {
        pointer = new Memory(data.length + 1);
        setData(data);
    }

    private void setData(byte[] data) {
        pointer.write(0, data, 0, data.length);
        pointer.setByte(data.length, (byte) 0);
    }

    /**
     * Acquires a pooled {@code NativeString} object if available, otherwise a new instance is
     * created.
     *
     * @param string the string value
     * @param encoding the charset encoding
     * @return a {@code NativeString} object
     */
    public static NativeString of(String string, Charset encoding) {
        byte[] data = string.getBytes(encoding);
        NativeString array = POOL.acquire();
        if (array != null && array.pointer.size() > data.length) {
            array.setData(data);
            return array;
        }
        return new NativeString(data);
    }

    /** Recycles this instance and return it back to the pool. */
    public void recycle() {
        POOL.recycle(this);
    }

    /**
     * Returns the peer pointer.
     *
     * @return the peer pointer
     */
    public Pointer getPointer() {
        return pointer;
    }
}
