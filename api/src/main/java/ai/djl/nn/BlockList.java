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
package ai.djl.nn;

import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.List;
import java.util.Map;

/** Represents a set of names and Blocks. */
public class BlockList extends PairList<String, Block> {

    /** Creates an empty {@code BlockList}. */
    public BlockList() {}

    /**
     * Constructs an empty {@code BlockList} with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public BlockList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified keys and values.
     *
     * @param keys the key list containing the elements to be placed into this {@code BlockList}
     * @param values the value list containing the elements to be placed into this {@code BlockList}
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public BlockList(List<String> keys, List<Block> values) {
        super(keys, values);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified list of Pairs.
     *
     * @param list the list containing the elements to be placed into this {@code BlockList}
     */
    public BlockList(List<Pair<String, Block>> list) {
        super(list);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified map.
     *
     * @param map the map containing keys and values
     */
    public BlockList(Map<String, Block> map) {
        super(map);
    }
}
