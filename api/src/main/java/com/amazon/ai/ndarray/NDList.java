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

    private List<NDArray> list;

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

    /**
     * Returns the NDArray at the specified position in this list.
     *
     * @param index index of the NDArray to return
     * @return the element at the specified position in this list
     * @throws IndexOutOfBoundsException if the index is out of range (<tt>index &lt; 0 || index
     *     &gt;= size()</tt>)
     */
    public NDArray get(int index) {
        return list.get(index);
    }

    /**
     * Appends the specified NDArray to the end of this list (optional operation).
     *
     * <p>Lists that support this operation may place limitations on what elements may be added to
     * this list. In particular, some lists will refuse to add null elements, and others will impose
     * restrictions on the type of elements that may be added. List classes should clearly specify
     * in their documentation any restrictions on what elements may be added.
     *
     * @param array element to be appended to this list
     * @return <tt>true</tt> if this array is successfully added
     * @throws UnsupportedOperationException if the <tt>add</tt> operation is not supported by this
     *     list
     * @throws ClassCastException if the class of the specified element prevents it from being added
     *     to this list
     * @throws NullPointerException if the specified element is null and this list does not permit
     *     null elements
     * @throws IllegalArgumentException if some property of this element prevents it from being
     *     added to this list
     */
    public boolean add(NDArray array) {
        return list.add(array);
    }

    @Override
    public Iterator<NDArray> iterator() {
        return list.iterator();
    }
}
