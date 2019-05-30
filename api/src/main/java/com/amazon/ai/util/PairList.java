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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

public class PairList<K, V> implements Iterable<Pair<K, V>> {

    private List<K> keys;
    private List<V> values;

    /** Constructs an empty <code>PairList</code> with an default initial capacity. */
    public PairList() {
        keys = new ArrayList<>();
        values = new ArrayList<>();
    }

    /**
     * Constructs an empty <code>PairList</code> with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public PairList(int initialCapacity) {
        keys = new ArrayList<>(initialCapacity);
        values = new ArrayList<>(initialCapacity);
    }

    /**
     * Constructs a <code>PairList</code> containing the elements of the specified keys and values.
     *
     * @param keys the key list whose elements are to be placed into this PairList
     * @param values the value list whose elements are to be placed into this PairList
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public PairList(List<K> keys, List<V> values) {
        if (keys.size() != values.size()) {
            throw new IllegalArgumentException("key value size mismatch.");
        }
        this.keys = keys;
        this.values = values;
    }

    public PairList(Map<K, V> map) {
        keys = new ArrayList<>(map.size());
        values = new ArrayList<>(map.size());
        for (Map.Entry<K, V> entry : map.entrySet()) {
            keys.add(entry.getKey());
            values.add(entry.getValue());
        }
    }

    public void add(K key, V value) {
        keys.add(key);
        values.add(value);
    }

    public void add(Pair<K, V> pair) {
        keys.add(pair.getKey());
        values.add(pair.getValue());
    }

    public int size() {
        return keys.size();
    }

    public Pair<K, V> get(int index) {
        return new Pair<>(keys.get(index), values.get(index));
    }

    public K keyAt(int index) {
        return keys.get(index);
    }

    public V valueAt(int index) {
        return values.get(index);
    }

    public K[] keys(K[] target) {
        return keys.toArray(target);
    }

    public V[] values(V[] target) {
        return values.toArray(target);
    }

    @Override
    public Iterator<Pair<K, V>> iterator() {
        return new Itr();
    }

    public Map<K, V> toMap() {
        return toMap(true);
    }

    public Map<K, V> toMap(boolean checkDuplicate) {
        Map<K, V> map = new HashMap<>();
        for (int i = 0, size = keys.size(); i < size; ++i) {
            if (map.put(keys.get(i), values.get(i)) != null && checkDuplicate) {
                throw new IllegalStateException("Duplicate keys: " + keys.get(i));
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
}
