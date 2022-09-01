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
package ai.djl.ndarray.index.full;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.index.dim.NDIndexAll;
import ai.djl.ndarray.index.dim.NDIndexElement;
import ai.djl.ndarray.index.dim.NDIndexTake;
import ai.djl.ndarray.types.Shape;

import java.util.Optional;

/** A simplified representation of a take-based {@link NDIndex}. */
public final class NDIndexFullTake {

    private NDArray indices;
    private int axis;

    /**
     * Constructs a new {@link NDIndexFullTake}.
     *
     * @param indices the indices to take
     * @param axis the axis to take at
     */
    private NDIndexFullTake(NDArray indices, int axis) {
        this.indices = indices;
        this.axis = axis;
    }

    /**
     * Returns (if possible) the {@link NDIndexFullTake} representation of an {@link NDIndex}.
     *
     * @param index the index to represent
     * @param target the shape of the array to index
     * @return the full take representation or nothing if it can't represent the index
     */
    public static Optional<NDIndexFullTake> fromIndex(NDIndex index, Shape target) {
        int axis = 0;
        NDIndexFullTake fullTake = null;
        for (NDIndexElement el : index.getIndices()) {
            if (el instanceof NDIndexAll) {
                axis++;
            } else if (el instanceof NDIndexTake) {
                if (fullTake != null) {
                    // Don't support multiple takes
                    throw new UnsupportedOperationException(
                            "Only one take per get is currently supported");
                }
                NDArray indexElem = ((NDIndexTake) el).getIndex();
                if (!indexElem.getShape().isRankOne()) {
                    throw new UnsupportedOperationException(
                            "Only rank-1 indexing array is supported for take");
                }
                fullTake = new NDIndexFullTake(indexElem, axis);
            } else {
                // Invalid dim for fullTake
                return Optional.empty();
            }
        }
        return Optional.ofNullable(fullTake);
    }

    /**
     * Returns the indices to take.
     *
     * @return the indices to take
     */
    public NDArray getIndices() {
        return indices;
    }

    /**
     * Returns the axis to take.
     *
     * @return the axis to take
     */
    public int getAxis() {
        return axis;
    }
}
