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
package com.amazon.ai.util;

import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

public class PairList<K, V> implements Iterable<Pair<K, V>> {

    @SuppressWarnings("rawtypes")
    private static final PairList EMPTY_LIST = new EmptyList();

    private final K[] keys;
    private final V[] values;

    public PairList(K[] keys, V[] values) {
        if (keys.length != values.length) {
            throw new IllegalArgumentException("key value size mismatch.");
        }
        this.keys = keys;
        this.values = values;
    }

    public int size() {
        return keys.length;
    }

    public Pair<K, V> get(int index) {
        return new Pair<>(keys[index], values[index]);
    }

    public K keyAt(int index) {
        return keys[index];
    }

    public V valueAt(int index) {
        return values[index];
    }

    public K[] keys(K[] target) {
        return keys;
    }

    public V[] values(V[] target) {
        return values;
    }

    @Override
    public Iterator<Pair<K, V>> iterator() {
        return new Itr();
    }

    @SuppressWarnings("unchecked")
    public static <S, T> PairList<S, T> fromList(List<S> keys, List<T> values) {
        int size = keys.size();
        if (size != values.size()) {
            throw new IllegalArgumentException("key value size mismatch.");
        }
        if (size == 0) {
            return EMPTY_LIST;
        }

        Class<?> keyType = keys.get(0).getClass();
        Class<?> valueType = values.get(0).getClass();
        S[] keyArray = (S[]) Array.newInstance(keyType, size);
        T[] valueArray = (T[]) Array.newInstance(valueType, size);
        for (int i = 0; i < size; ++i) {
            keyArray[i] = keys.get(i);
            valueArray[i] = values.get(i);
        }
        return new PairList<>(keyArray, valueArray);
    }

    @SuppressWarnings("unchecked")
    public static <S, T> PairList<S, T> fromMap(Map<S, T> map) {
        if (map == null || map.isEmpty()) {
            return EMPTY_LIST;
        }

        Set<Map.Entry<S, T>> entries = map.entrySet();
        Map.Entry<S, T> first = entries.iterator().next();
        Class<?> keyType = first.getKey().getClass();
        Class<?> valueType = first.getKey().getClass();

        S[] keys = (S[]) Array.newInstance(keyType, map.size());
        T[] values = (T[]) Array.newInstance(valueType, map.size());
        int i = 0;
        for (Map.Entry<S, T> entry : entries) {
            keys[i] = entry.getKey();
            values[i] = entry.getValue();
            ++i;
        }
        return new PairList<>(keys, values);
    }

    public Map<K, V> toMap() {
        return toMap(true);
    }

    public Map<K, V> toMap(boolean checkDuplicate) {
        Map<K, V> map = new HashMap<>();
        for (int i = 0; i < keys.length; ++i) {
            if (map.put(keys[i], values[i]) != null && checkDuplicate) {
                throw new IllegalStateException("Duplicate keys: " + keys[i]);
            }
        }
        return map;
    }

    private class Itr implements Iterator<Pair<K, V>> {
        private int cursor;

        Itr() {}

        public boolean hasNext() {
            return cursor < size();
        }

        public Pair<K, V> next() {
            if (cursor >= size()) {
                throw new NoSuchElementException();
            }

            return get(cursor++);
        }
    }

    private static final class EmptyList<S, T> extends PairList<S, T> {

        @SuppressWarnings("unchecked")
        public EmptyList() {
            super((S[]) new Object[0], (T[]) new Object[0]);
        }

        @Override
        public S[] keys(S[] target) {
            if (target.length > 0) {
                target[0] = null;
            }
            return target;
        }

        @Override
        public T[] values(T[] target) {
            if (target.length > 0) {
                target[0] = null;
            }
            return target;
        }
    }
}
