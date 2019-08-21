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
package software.amazon.ai;

import java.util.List;
import java.util.Map;
import software.amazon.ai.nn.Block;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

/** Represents a set of names and Blocks. */
public class BlockList extends PairList<String, Block> {

    public BlockList() {}

    public BlockList(int initialCapacity) {
        super(initialCapacity);
    }

    public BlockList(List<String> keys, List<Block> values) {
        super(keys, values);
    }

    public BlockList(List<Pair<String, Block>> list) {
        super(list);
    }

    public BlockList(Map<String, Block> map) {
        super(map);
    }
}
