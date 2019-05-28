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
package com.amazon.ai.ndarray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class NDList implements Iterable<NDArray> {

    protected List<NDArray> list;

    /** Constructs an empty list. */
    public NDList() {
        this.list = new ArrayList<>();
    }

    /**
     * Constructs an empty NDList with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public NDList(int initialCapacity) {
        this.list = new ArrayList<>(initialCapacity);
    }

    public NDList(NDArray... arrays) {
        this.list = Arrays.asList(arrays);
    }

    public NDList(NDList arrays) {
        this.list = arrays.list;
    }

    public NDArray[] toArray() {
        return list.toArray(new NDArray[0]);
    }

    /**
     * Returns the number of NDArray in this list. If this list contains more than
     * <tt>Integer.MAX_VALUE</tt> NDArrays, returns <tt>Integer.MAX_VALUE</tt>.
     *
     * @return the number of NDArrays in this list
     */
    public int size() {
        return list.size();
    }

    /**
     * Returns the NDArray at the specified position in this list.
     *
     * @param index index of the NDArray to return
     * @return the NDArray at the specified position in this list
     * @throws IndexOutOfBoundsException if the index is out of range (<tt>index &lt; 0 || index
     *     &gt;= size()</tt>)
     */
    public NDArray get(int index) {
        return list.get(index);
    }

    /**
     * Appends the specified NDArray to the end of this list (optional operation).
     *
     * <p>Lists that support this operation may place limitations on what NDArrays may be added to
     * this list. In particular, some lists will refuse to add null NDArrays, and others will impose
     * restrictions on the type of NDArrays that may be added. List classes should clearly specify
     * in their documentation any restrictions on what NDArrays may be added.
     *
     * @param array NDArray to be appended to this list
     * @return <tt>true</tt> if this array is successfully added
     * @throws UnsupportedOperationException if the <tt>add</tt> operation is not supported by this
     *     list
     * @throws ClassCastException if the class of the specified NDArray prevents it from being added
     *     to this list
     * @throws NullPointerException if the specified NDArray is null and this list does not permit
     *     null NDArrays
     * @throws IllegalArgumentException if some property of this NDArray prevents it from being
     *     added to this list
     */
    public boolean add(NDArray array) {
        return list.add(array);
    }

    /**
     * Appends all of the NDArray in the specified collection to the end of this list, in the order
     * that they are returned by the specified collection's iterator (optional operation). The
     * behavior of this operation is undefined if the specified collection is modified while the
     * operation is in progress. (Note that this will occur if the specified collection is this
     * list, and it's nonempty.)
     *
     * @param other NDList containing NDArray to be added to this list
     * @return <tt>true</tt> if this list changed as a result of the call
     * @throws UnsupportedOperationException if the <tt>addAll</tt> operation is not supported by
     *     this list
     * @throws ClassCastException if the class of a NDArray of the specified collection prevents it
     *     from being added to this list
     * @throws NullPointerException if the specified collection contains one or more null NDArray
     *     and this list does not permit null NDArrays, or if the specified collection is null
     * @throws IllegalArgumentException if some property of an NDArray of the specified collection
     *     prevents it from being added to this list
     */
    public void addAll(NDList other) {
        list.addAll(other.list);
    }

    /**
     * Returns an iterator over the NDArrays in this list in proper sequence.
     *
     * @return an iterator over the NDArrays in this list in proper sequence
     */
    @Override
    public Iterator<NDArray> iterator() {
        return list.iterator();
    }

    public void waitToRead() {
        for (NDArray array : this) {
            array.waitToRead();
        }
    }
}
