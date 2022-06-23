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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;

import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Utility class that provides some useful blocks. */
public final class Blocks {

    private Blocks() {}

    /**
     * Inflates the {@link ai.djl.ndarray.NDArray} provided as input to a 2-D {@link
     * ai.djl.ndarray.NDArray} of shape (batch, size).
     *
     * @param array a array to be flattened
     * @return a {@link NDList} that contains the inflated {@link ai.djl.ndarray.NDArray}
     */
    public static NDArray batchFlatten(NDArray array) {
        long batch = array.size(0);
        if (batch == 0) {
            // calculate the size of second dimension manually as using -1 would not work here
            return array.reshape(batch, array.getShape().slice(1).size());
        }
        return array.reshape(batch, -1);
    }

    /**
     * Inflates the {@link ai.djl.ndarray.NDArray} provided as input to a 2-D {@link
     * ai.djl.ndarray.NDArray} of shape (batch, size).
     *
     * @param array a array to be flattened
     * @param size the input size
     * @return a {@link NDList} that contains the inflated {@link ai.djl.ndarray.NDArray}
     * @throws IndexOutOfBoundsException if the input {@link NDList} has more than one {@link
     *     ai.djl.ndarray.NDArray}
     */
    public static NDArray batchFlatten(NDArray array, long size) {
        return array.reshape(-1, size);
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     * batchFlatten} method.
     *
     * @return a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     *     batchFlatten} method
     */
    public static Block batchFlattenBlock() {
        return LambdaBlock.singleton(Blocks::batchFlatten, "batchFlatten");
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     * batchFlatten} method. The size of input to the block returned must be batch_size * size.
     *
     * @param size the expected size of each input
     * @return a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     *     batchFlatten} method
     */
    public static Block batchFlattenBlock(long size) {
        return LambdaBlock.singleton(array -> batchFlatten(array, size), "batchFlatten");
    }

    /**
     * Creates a {@link LambdaBlock} that performs the identity function.
     *
     * @return an identity {@link Block}
     */
    public static Block identityBlock() {
        return new LambdaBlock(x -> x, "identity");
    }

    /**
     * Returns a string representation of the passed {@link Block} describing the input axes, output
     * axes, and the block's children.
     *
     * @param block the block to describe
     * @param blockName the name to be used for the passed block, or <code>null</code> if its class
     *     name is to be used
     * @param beginAxis skips all axes before this axis; use <code>0</code> to print all axes and
     *     <code>1</code> to skip the batch axis.
     * @return the string representation
     */
    public static String describe(Block block, String blockName, int beginAxis) {
        Shape[] inputShapes = block.isInitialized() ? block.getInputShapes() : null;
        Shape[] outputShapes = inputShapes != null ? block.getOutputShapes(inputShapes) : null;
        StringBuilder sb = new StringBuilder(200);
        if (block instanceof LambdaBlock
                && !LambdaBlock.DEFAULT_NAME.equals(((LambdaBlock) block).getName())) {
            sb.append(((LambdaBlock) block).getName());
        } else if (blockName != null) {
            sb.append(blockName);
        } else {
            sb.append(block.getClass().getSimpleName());
        }
        if (inputShapes != null) {
            sb.append(
                    Stream.of(inputShapes)
                            .map(shape -> shape.slice(beginAxis).toString())
                            .collect(Collectors.joining("+")));
        }
        if (!block.getChildren().isEmpty()) {
            sb.append(" {\n");
            for (Pair<String, Block> pair : block.getChildren()) {
                String child = describe(pair.getValue(), pair.getKey().substring(2), beginAxis);
                sb.append(child.replaceAll("(?m)^", "\t")).append('\n');
            }
            sb.append('}');
        }
        if (outputShapes != null) {
            sb.append(" -> ");
            sb.append(
                    Stream.of(outputShapes)
                            .map(shape -> shape.slice(beginAxis).toString())
                            .collect(Collectors.joining("+")));
        }
        return sb.toString();
    }
}
