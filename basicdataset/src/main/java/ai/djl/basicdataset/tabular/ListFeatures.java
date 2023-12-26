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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/** An extension of the {@link ArrayList} for use in the {@link TabularTranslator}. */
public class ListFeatures extends ArrayList<String> {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a {@code ListFeatures} instance.
     *
     * @see ArrayList#ArrayList()
     */
    public ListFeatures() {}

    /**
     * Constructs a {@code ListFeatures} instance.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     * @see ArrayList#ArrayList(int)
     */
    public ListFeatures(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code ListFeatures} instance from a source list.
     *
     * @param source the source list
     */
    @SuppressWarnings("this-escape")
    public ListFeatures(List<String> source) {
        super(source.size());
        addAll(source);
    }

    /**
     * Constructs a {@code ListFeatures} instance.
     *
     * @param c the collection whose elements are to be placed into this list
     * @throws NullPointerException if the specified collection is null
     * @see ArrayList#ArrayList(Collection)
     */
    public ListFeatures(Collection<? extends String> c) {
        super(c);
    }
}
