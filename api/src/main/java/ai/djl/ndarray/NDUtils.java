/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray;

import ai.djl.ndarray.types.Shape;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/** A class containing utility methods for NDArray operations. */
public final class NDUtils {

    private NDUtils() {}

    /**
     * Get {@link Shape} of the empty {@link NDArray} after applying reduction operations.
     *
     * @param shape input shape
     * @param axis axis to apply reduction
     * @return the result {@link Shape}
     */
    public static Shape getShapeFromEmptyNDArrayForReductionOp(Shape shape, int axis) {
        final long[] shapeArr = shape.getShape();
        if (shapeArr[axis] == 0) {
            throw new IllegalArgumentException("attempt to apply reduction of an empty NDArray");
        }
        long[] newShape =
                IntStream.range(0, shapeArr.length)
                        .filter(i -> i != axis)
                        .mapToLong(i -> shapeArr[i])
                        .toArray();
        return new Shape(newShape);
    }

    /**
     * Check two criteria of concat input: 1. no scalar 2. dimensions of all the array must be the
     * same.
     *
     * @param list input {@link NDList}
     */
    public static void checkConcatInput(NDList list) {
        NDArray[] arrays = list.toArray(new NDArray[0]);
        if (Stream.of(arrays).allMatch(array -> array.getShape().dimension() == 0)) {
            throw new IllegalArgumentException(
                    "scalar(zero-dimensional) arrays cannot be concatenated");
        }
        int dimension = arrays[0].getShape().dimension();
        for (int i = 1; i < arrays.length; i++) {
            if (arrays[i].getShape().dimension() != dimension) {
                throw new IllegalArgumentException(
                        "all the input arrays must have same number of dimensions, but the array at index 0 has "
                                + dimension
                                + " dimension(s) and the array at index "
                                + i
                                + " has "
                                + arrays[i].getShape().dimension()
                                + " dimension(s)");
            }
        }
    }
}
