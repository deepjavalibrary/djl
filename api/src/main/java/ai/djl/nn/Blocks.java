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

    private static NDList flatten(NDList arrays, long batch, long size) {
        return new NDList(arrays.singletonOrThrow().reshape(batch, size));
    }

    public static Block flattenBlock() {
        return new LambdaBlock(arrays -> flatten(arrays));
    }

    public static Block flattenBlock(long size) {
        return flattenBlock(-1, size);
    }

    public static Block flattenBlock(long batch, long size) {
        return new LambdaBlock(arrays -> flatten(arrays, batch, size));
    }
}
