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
package software.amazon.ai.ndarray;

import java.util.Iterator;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

/**
 * An {@code NDList} represents a sequence of {@link NDArray}s with names.
 *
 * <p>Each {@link NDArray} in this list can optionally have a name. You can use the name to look up
 * NDArray in the NDList.
 */
public class NDList implements Iterable<Pair<String, NDArray>> {

    protected PairList<String, NDArray> list;

    /** Constructs an empty NDList. */
    public NDList() {
        this.list = new PairList<>();
    }

    /**
     * Constructs an empty NDList with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public NDList(int initialCapacity) {
        this.list = new PairList<>(initialCapacity);
    }

    /**
     * Constructs and initiates an NDList with the specified {@link NDArray}s.
     *
     * @param arrays {@link NDArray}s
     */
    public NDList(NDArray... arrays) {
        this();
        for (NDArray array : arrays) {
            list.add(null, array);
        }
    }

    /**
     * Returns an array of {@link NDArray} in the same order as in the NDList.
     *
     * <p>The returning array will lose the name information in the NDList.
     *
     * @return an array of {@link NDArray}
     */
    public NDArray[] toArray() {
        return list.valueArray(new NDArray[0]);
    }

    /**
     * Returns the number of NDArrays in this NDList.
     *
     * <p>If this list contains more than {@code Integer.MAX_VALUE} NDArrays, returns {@code
     * Integer.MAX_VALUE}.
     *
     * @return the number of NDArrays in this NDList
     */
    public int size() {
        return list.size();
    }

    /**
     * Removes the first occurrence of the specified element from this NDList if it is present
     * (optional operation).
     *
     * <p>If this list does not contain the element, it is unchanged. More formally, removes the
     * element with the lowest index {@code i} such that {@code
     * (o==null&nbsp;?&nbsp;get(i)==null&nbsp;:&nbsp;o.equals(get(i)))} (if such an element exists).
     *
     * @param name name of the NDArray to be removed from this NDList, if present
     * @return the element which got removed
     * @throws UnsupportedOperationException if the {@code NDList} is read only
     */
    public NDArray remove(String name) {
        return list.remove(name);
    }

    /**
     * Returns {@code true} if this NDList contains NDArray with the specified name.
     *
     * @param name name of the NDArray to be removed from this NDList, if present
     * @return {@code true} if this list contains the specified element
     */
    public boolean contains(String name) {
        return list.contains(name);
    }

    /**
     * Returns the NDArray at the specified position in this list.
     *
     * @param index index of the NDArray to return
     * @return the NDArray at the specified position in this list
     * @throws IndexOutOfBoundsException if the index is out of range ({@code index &lt; 0 || index
     *     &gt;= size()})
     */
    public NDArray get(int index) {
        return list.valueAt(index);
    }

    /**
     * Returns the head index of the NDList.
     *
     * @return the head NDArray
     * @throws IndexOutOfBoundsException if the index is out of range ({@code index &lt; 0 || index
     *     &gt;= size()})
     */
    public NDArray head() {
        return list.valueAt(0);
    }

    /**
     * Appends the specified NDArray to the end of this NDList.
     *
     * @param array NDArray to be appended to this list
     * @throws UnsupportedOperationException if this NDList is read only
     * @see NDList#add(String, NDArray)
     */
    public void add(NDArray array) {
        list.add(null, array);
    }

    /**
     * Appends the named NDArray to the end of this NDList.
     *
     * @param name optional name of the {@link NDArray}
     * @param array NDArray to be appended to this list
     * @throws UnsupportedOperationException if this NDList is read only
     */
    public void add(String name, NDArray array) {
        list.add(name, array);
    }

    /**
     * Appends all of the NDArrays in the specified NDList to the end of this NDList, in the order
     * that they are returned by the specified DNList's iterator.
     *
     * @param other NDList containing NDArray to be added to this list
     * @throws UnsupportedOperationException if this NDList is read only
     */
    public void addAll(NDList other) {
        for (Pair<String, NDArray> pair : other) {
            list.add(pair);
        }
    }

    public void attach(NDManager manager) {
        for (NDArray array : list.values()) {
            array.attach(manager);
        }
    }

    public void detach() {
        for (NDArray array : list.values()) {
            array.detach();
        }
    }

    /**
     * Returns an iterator over the NDArrays in this list in proper sequence.
     *
     * @return an iterator over the NDArrays in this list in proper sequence
     */
    @Override
    public Iterator<Pair<String, NDArray>> iterator() {
        return list.iterator();
    }
}
