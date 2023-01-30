/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.basicdataset.tabular;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** An extension of the {@link ConcurrentHashMap} for use in the {@link TabularTranslator}. */
public class MapFeatures extends ConcurrentHashMap<String, String> {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a {@code MapFeatures} instance.
     *
     * @see ConcurrentHashMap#ConcurrentHashMap()
     */
    public MapFeatures() {}

    /**
     * Constructs a {@code MapFeatures} instance.
     *
     * @param initialCapacity The implementation performs internal sizing to accommodate this many
     *     elements.
     * @throws IllegalArgumentException if the initial capacity of elements is negative
     * @see ConcurrentHashMap#ConcurrentHashMap(int)
     */
    public MapFeatures(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code MapFeatures} instance.
     *
     * @param m the map
     * @see ConcurrentHashMap#ConcurrentHashMap(Map)
     */
    public MapFeatures(Map<? extends String, ? extends String> m) {
        super(m);
    }

    /**
     * Constructs a {@code MapFeatures} instance.
     *
     * @param initialCapacity the initial capacity. The implementation performs internal sizing to
     *     accommodate this many elements, given the specified load factor.
     * @param loadFactor the load factor (table density) for establishing the initial table size
     * @throws IllegalArgumentException if the initial capacity of elements is negative or the load
     *     factor is nonpositive
     * @see ConcurrentHashMap#ConcurrentHashMap(int, float)
     */
    public MapFeatures(int initialCapacity, float loadFactor) {
        super(initialCapacity, loadFactor);
    }

    /**
     * Constructs a {@link MapFeatures}.
     *
     * @param initialCapacity the initial capacity. The implementation performs internal sizing to
     *     accommodate this many elements, given the specified load factor.
     * @param loadFactor the load factor (table density) for establishing the initial table size
     * @param concurrencyLevel the estimated number of concurrently updating threads. The
     *     implementation may use this value as a sizing hint.
     * @throws IllegalArgumentException if the initial capacity is negative or the load factor or
     *     concurrencyLevel are nonpositive
     * @see ConcurrentHashMap#ConcurrentHashMap(int, float, int)
     */
    public MapFeatures(int initialCapacity, float loadFactor, int concurrencyLevel) {
        super(initialCapacity, loadFactor, concurrencyLevel);
    }

    /**
     * Creates a {@link MapFeatures} from a source list.
     *
     * @param source the source list
     * @return a new {@link MapFeatures}
     */
    public static MapFeatures fromMap(Map<String, String> source) {
        MapFeatures map = new MapFeatures(source.size());
        map.putAll(source);
        return map;
    }
}
