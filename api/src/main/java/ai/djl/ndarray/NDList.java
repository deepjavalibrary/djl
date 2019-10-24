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
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.Iterator;

/**
 * An {@code NDList} represents a sequence of {@link NDArray}s with names.
 *
 * <p>Each {@link NDArray} in this list can optionally have a name. You can use the name to look up
 * NDArray in the NDList.
 */
public class NDList implements Iterable<Pair<String, NDArray>>, AutoCloseable {

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
     * Returns true if size() is 0.
     *
     * @return true if size() is 0, otherwise false.
     */
    public boolean isEmpty() {
        return size() == 0;
    }

    /**
     * Removes the first occurrence of the specified element from this NDList if it is present.
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
     * Removes the index of the specified element from this NDList if it is present.
     *
     * @param index the index of the element to remove
     * @return the element which got removed
     */
    public NDArray remove(int index) {
        return list.remove(index);
    }

    /**
     * Returns a view of the portion of this NDList between the specified <tt>fromIndex</tt>.
     * inclusive, and <tt>toIndex</tt>, exclusive.
     *
     * @param fromIndex the start index (inclusive)
     * @param toIndex the end index (exclusive)
     * @return a view of the portion of this NDList
     */
    public NDList subList(int fromIndex, int toIndex) {
        NDList subList = new NDList();
        subList.addAll(list.subList(fromIndex, toIndex));
        return subList;
    }

    /**
     * Returns a view of the portion of this NDList between the specified <tt>fromIndex</tt>.
     * inclusive, and to the end
     *
     * @param fromIndex the start index (inclusive)
     * @return a view of the portion of this NDList
     */
    public NDList subList(int fromIndex) {
        return subList(fromIndex, size());
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
     * Get NDArray with Tag.
     *
     * @param index numeric index to get
     * @return tag and ndarray
     */
    public Pair<String, NDArray> getWithTag(int index) {
        return list.get(index);
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
     * Returns the only element if this is a singleton NDList or throws an exception if multiple
     * elements.
     *
     * @return the head NDArray
     * @throws IndexOutOfBoundsException if the list does not contain exactly one element
     */
    public NDArray singletonOrThrow() {
        if (size() != 1) {
            throw new IndexOutOfBoundsException(
                    "Incorrect number of elements in NDList.singletonOrThrow: Expected 1 and was "
                            + size());
        }
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

    /**
     * Appends all of the pairs in the specified PairList to the end of this NDList, in the order
     * that they are returned by the specified PairList's iterator.
     *
     * @param other PairList containing String NDArray pair to be added to this list
     * @throws UnsupportedOperationException if this NDList is read only
     */
    public void addAll(PairList<String, NDArray> other) {
        list.addAll(other);
    }

    /**
     * Converts all the {@code NDArray} in {@code NDList} to a different {@link Device}.
     *
     * @param device {@link Device} to be set
     * @param copy set {@code true} if you want to return a copy of the underlying NDArray.
     * @return a new {@code NDList} with the NDArrays on specified {@link Device}
     */
    public NDList asInDevice(Device device, boolean copy) {
        NDList newNDList = new NDList(size());
        for (Pair<String, NDArray> pair : list) {
            NDArray array = pair.getValue().asInDevice(device, copy);
            newNDList.add(pair.getKey(), array);
        }

        return newNDList;
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

    @Override
    public void close() {
        for (NDArray array : list.values()) {
            array.close();
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(20);
        builder.append("NDList size: ").append(size()).append('\n');
        for (int i = 0; i < list.size(); i++) {
            Pair<String, NDArray> pair = list.get(i);
            builder.append(i).append(' ');
            if (pair.getKey() != null) {
                builder.append(pair.getKey());
            }
            builder.append(": ")
                    .append(pair.getValue().getShape())
                    .append(' ')
                    .append(pair.getValue().getDataType())
                    .append('\n');
        }
        return builder.toString();
    }
}
