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

import ai.djl.ndarray.NDList;

public final class Blocks {

    private Blocks() {}

    private static NDList flatten(NDList arrays) {
        long batch = arrays.head().size(0);
        return flatten(arrays, batch, -1);
    }

    /**
     * Flattens the only {@link ai.djl.ndarray.NDArray} in the input to a 2-D {@link
     * ai.djl.ndarray.NDArray} of shape (batch, size).
     *
     * @param array a singleton {@link NDList}
     * @param batch the batch size
     * @param size the size of the flattened array
     * @return a singleton {@link NDList} that contains the flattened {@link ai.djl.ndarray.NDArray}
     * @throws IndexOutOfBoundsException if the input {@link NDList} has more than one {@link
     *     ai.djl.ndarray.NDArray}
     */
    private static NDList flatten(NDList array, long batch, long size) {
        return new NDList(array.singletonOrThrow().reshape(batch, size));
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #flatten(NDList) flatten}
     * method.
     *
     * @return a {@link Block} whose forward function applies the {@link #flatten(NDList) flatten}
     *     method
     */
    public static Block flattenBlock() {
        return new LambdaBlock(arrays -> flatten(arrays));
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #flatten(NDList) flatten}
     * method. The size of input to the block returned must be batch_size * size.
     *
     * @param size the expected size of each input
     * @return a {@link Block} whose forward function applies the {@link #flatten(NDList) flatten}
     *     method
     */
    public static Block flattenBlock(long size) {
        return flattenBlock(-1, size);
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #flatten(NDList) flatten}
     * method. The size of input to the block returned must be batch * size.
     *
     * @param batch the batch size
     * @param size the expected size of each input
     * @return a {@link Block} whose forward function applies the {@link #flatten(NDList) flatten}
     *     method
     */
    public static Block flattenBlock(long batch, long size) {
        return new LambdaBlock(arrays -> flatten(arrays, batch, size));
    }
}
