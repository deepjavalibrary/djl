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
package ai.djl.paddlepaddle.jna;

import com.sun.jna.Function;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/**
 * An abstraction for a native pointer array data type ({@code void**}).
 *
 * @see Pointer
 * @see com.sun.jna.ptr.PointerByReference
 * @see Function
 */
@SuppressWarnings("checkstyle:EqualsHashCode")
final class PointerArray extends Memory {

    private int length;

    /**
     * Constructs a {@link Memory} buffer PointerArray given the Pointers to include in it.
     *
     * @param arg the pointers to include in the array
     */
    public PointerArray(Pointer... arg) {
        super(Native.POINTER_SIZE * (arg.length + 1));
        length = arg.length;
        setPointers(arg);
    }

    private void setPointers(Pointer[] pointers) {
        for (int i = 0; i < pointers.length; i++) {
            setPointer(i * Native.POINTER_SIZE, pointers[i]);
        }
        setPointer(Native.POINTER_SIZE * length, null);
    }

    /**
     * Returns the number of array elements.
     *
     * @return the number of array elements
     */
    public int numElements() {
        return length;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        return o == this;
    }
}
