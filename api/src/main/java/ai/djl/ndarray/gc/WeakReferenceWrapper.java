/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray.gc;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;

/**
 * {@code WeakReferenceWrapper<K, V>} extends a {@link WeakReference}. It uses an object of type K
 * as the referent has an object property of type V.
 */
public class WeakReferenceWrapper<K, V> extends WeakReference<Object> {
    V value;

    WeakReferenceWrapper(K key, V value, ReferenceQueue<Object> queue) {
        super(key, queue);
        this.value = value;
    }

    /**
     * Returns the value.
     *
     * @return the value
     */
    public V getValue() {
        return value;
    }
}
