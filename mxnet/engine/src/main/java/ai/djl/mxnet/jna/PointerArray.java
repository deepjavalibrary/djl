/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import com.sun.jna.Native;
import com.sun.jna.Pointer;

public class PointerArray extends Memory {

    private int length;

    public PointerArray(Pointer... arg) {
        super(Native.POINTER_SIZE * (arg.length + 1));
        length = arg.length;
        for (int i = 0; i < arg.length; i++) {
            setPointer(i * Native.POINTER_SIZE, arg[i]);
        }
        setPointer(Native.POINTER_SIZE * arg.length, null);
    }

    public int numElements() {
        return length;
    }

    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        }
        if (o == null) {
            return false;
        }
        return (o instanceof Pointer)
                && (((PointerArray) o).numElements() == numElements())
                && super.equals(o);
    }

    @Override
    public int hashCode() {
        return super.hashCode() ^ this.numElements();
    }
}
