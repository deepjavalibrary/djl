/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package com.sun.jna;

/** {@code PointerProxy} is a Pointer wrapper that can access to the peer value of the pointer. */
public class PointerProxy extends Pointer {

    /**
     * Creates from native pointer. Don't use this unless you know what you're doing.
     *
     * @param ptr the target pointer
     */
    public PointerProxy(Pointer ptr) {
        super(ptr.peer);
    }

    /**
     * Gets peer value from the {@link Pointer}. /
     *
     * @return peer value in long
     */
    public long getPeer() {
        return peer;
    }
}
