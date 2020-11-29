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
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/** An abstraction for a native string array data type ({@code char**}). */
@SuppressWarnings("checkstyle:EqualsHashCode")
class StringArray extends Memory {

    private static final Charset ENCODING = Native.DEFAULT_CHARSET;

    /** Hold all {@code NativeString}, avoid be GCed. */
    private List<NativeString> natives; // NOPMD

    /**
     * Create a native array of strings.
     *
     * @param strings the strings
     */
    public StringArray(String[] strings) {
        super((strings.length + 1) * Native.POINTER_SIZE);
        natives = new ArrayList<>();
        for (int i = 0; i < strings.length; i++) {
            Pointer p = null;
            if (strings[i] != null) {
                NativeString ns = new NativeString(strings[i], ENCODING);
                natives.add(ns);
                p = ns.getPointer();
            }
            setPointer(Native.POINTER_SIZE * i, p);
        }
        setPointer(Native.POINTER_SIZE * strings.length, null);
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        return this == o;
    }
}
