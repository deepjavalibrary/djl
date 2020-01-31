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
package ai.djl.pytorch.jni;

/**
 * An abstraction for a native pointer data type. A Pointer instance represents, on the Java side, a
 * native pointer. The native pointer could be any <em>type</em> of native pointer.
 */
public class Pointer {

    private final long peer;

    /**
     * Creates an instance of {@link Pointer}.
     *
     * @param peer the native peer of the pointer
     */
    public Pointer(long peer) {
        this.peer = peer;
    }

    /**
     * Returns the native peer of the pointer address.
     *
     * @return the native peer of the pointer address
     */
    public long getValue() {
        return peer;
    }
}
