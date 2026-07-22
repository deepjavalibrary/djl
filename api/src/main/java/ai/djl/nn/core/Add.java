/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn.core;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.nn.ParallelBlock;

import java.util.Collections;
import java.util.List;

/**
 * {@code Add} is a {@link Block} whose children form a parallel branch in the network and are
 * combined by {@link NDArrays#add(NDArray...)} to produce a single output.
 *
 * <p>{@code Add} has no direct parameters.
 */
public class Add extends ParallelBlock {

    /**
     * Creates a block whose branches are combined to form a single output by {@link
     * NDArrays#add(NDArray...)}.
     */
    public Add() {
        this(Collections.emptyList());
    }

    /**
     * Creates a block whose branches are formed by each block in the list of blocks, and are
     * combined by {@link NDArrays#add(NDArray...)} to form a single output.
     *
     * @param blocks the blocks that form each of the parallel branches
     */
    public Add(List<Block> blocks) {
        super(
                list -> {
                    NDArray[] arrays = list.stream().map(NDList::head).toArray(NDArray[]::new);
                    return new NDList(NDArrays.add(arrays));
                },
                blocks);
    }
}
