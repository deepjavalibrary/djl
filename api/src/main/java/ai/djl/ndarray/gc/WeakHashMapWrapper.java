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

import ai.djl.ndarray.NDArray;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.reflect.Proxy;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

/**
 * {@code WeakHashMapWrapper<K, V>} wraps a {@link WeakHashMap}. It has a {@link ReferenceQueue} to
 * get informed by the garbage collector. On method invocations it looks for messages from the
 * garbage collector and removes the corresponding entries.
 */
public class WeakHashMapWrapper<K, V> implements Map<K, V> {

    private final WeakHashMap<K, V> map = new WeakHashMap<>();
    private final ReferenceQueue<Object> queue = new ReferenceQueue<>();

    private final Set<WeakReferenceWrapper<K, V>> weakReferenceWrapperSet = new HashSet<>();

    private void checkQueue() {
        for (Reference<?> ref; (ref = queue.poll()) != null; ) {
            synchronized (queue) {
                @SuppressWarnings("unchecked")
                WeakReferenceWrapper<K, V> ref2 = (WeakReferenceWrapper<K, V>) ref;
                V value = ref2.getValue();
                if (value instanceof NDArray) { // just as one example
                    ((NDArray) value).close();
                    weakReferenceWrapperSet.remove(ref2);
                }
            }
        }
    }

    // implement all methods of Map<K,V> interface by calling corresponding methods of
    // WeakHashMap<K,V> instance map

    /** {@inheritDoc} */
    @Override
    public int size() {
        checkQueue();
        return map.size();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        checkQueue();
        return map.isEmpty();
    }

    /** {@inheritDoc} */
    @Override
    public boolean containsKey(Object key) {
        checkQueue();
        return map.containsKey(key);
    }

    /** {@inheritDoc} */
    @Override
    public boolean containsValue(Object value) {
        checkQueue();
        return map.containsValue(value);
    }

    /** {@inheritDoc} */
    @Override
    public V get(Object key) {
        checkQueue();
        return map.get(key);
    }

    /** {@inheritDoc} */
    @Override
    public V put(K key, V value) {
        if (value instanceof Proxy) {
            throw new IllegalArgumentException(
                    "Proxy is not supported to be stored as value here.");
        }
        weakReferenceWrapperSet.add(new WeakReferenceWrapper<K, V>(key, value, queue));
        return map.put(key, value);
    }

    /** {@inheritDoc} */
    @Override
    public V remove(Object key) {
        checkQueue();
        return map.remove(key);
    }

    /** {@inheritDoc} */
    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        checkQueue();
        map.putAll(m);
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        checkQueue();
        map.clear();
    }

    /** {@inheritDoc} */
    @Override
    public Set<K> keySet() {
        checkQueue();
        return map.keySet();
    }

    /** {@inheritDoc} */
    @Override
    public Collection<V> values() {
        checkQueue();
        return map.values();
    }

    /** {@inheritDoc} */
    @Override
    public Set<Entry<K, V>> entrySet() {
        checkQueue();
        return map.entrySet();
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        checkQueue();
        return map.equals(o);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return map.hashCode();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return map.toString();
    }
}
