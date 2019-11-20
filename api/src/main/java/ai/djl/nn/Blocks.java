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

/** Utility class that provides some useful blocks. */
public final class Blocks {

    private Blocks() {}

    private static NDList batchFlatten(NDList arrays) {
        long batch = arrays.head().size(0);
        return batchFlatten(arrays, batch, -1);
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
    private static NDList batchFlatten(NDList array, long batch, long size) {
        return new NDList(array.singletonOrThrow().reshape(batch, size));
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #batchFlatten(NDList)
     * batchFlatten} method.
     *
     * @return a {@link Block} whose forward function applies the {@link #batchFlatten(NDList)
     *     batchFlatten} method
     */
    public static Block batchFlattenBlock() {
        return new LambdaBlock(arrays -> batchFlatten(arrays));
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #batchFlatten(NDList)
     * batchFlatten} method. The size of input to the block returned must be batch_size * size.
     *
     * @param size the expected size of each input
     * @return a {@link Block} whose forward function applies the {@link #batchFlatten(NDList)
     *     batchFlatten} method
     */
    public static Block batchFlattenBlock(long size) {
        return batchFlattenBlock(-1, size);
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #batchFlatten(NDList)
     * batchFlatten} method. The size of input to the block returned must be batch * size.
     *
     * @param batch the batch size
     * @param size the expected size of each input
     * @return a {@link Block} whose forward function applies the {@link #batchFlatten(NDList)
     *     batchFlatten} method
     */
    public static Block batchFlattenBlock(long batch, long size) {
        return new LambdaBlock(arrays -> batchFlatten(arrays, batch, size));
    }

    /**
     * Creates a {@link LambdaBlock} that performs the identity function.
     *
     * @return an identity {@link Block}
     */
    public static Block identityBlock() {
        return new LambdaBlock(x -> x);
    }
}
