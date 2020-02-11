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
}
