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
package ai.djl.ndarray.index;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.index.dim.NDIndexBooleans;
import ai.djl.ndarray.index.dim.NDIndexElement;
import ai.djl.ndarray.index.full.NDIndexFullPick;
import ai.djl.ndarray.index.full.NDIndexFullSlice;
import java.util.List;
import java.util.Optional;

/** A helper class for {@link NDArray} implementations for operations with an {@link NDIndex}. */
public abstract class NDArrayIndexer {

    /**
     * Returns a subarray by picking the elements.
     *
     * @param array the array to get from
     * @param fullPick the elements to pick
     * @return the subArray
     */
    public abstract NDArray get(NDArray array, NDIndexFullPick fullPick);

    /**
     * Returns a subarray at the slice.
     *
     * @param array the array to get from
     * @param fullSlice the fullSlice index of the array
     * @return the subArray
     */
    public abstract NDArray get(NDArray array, NDIndexFullSlice fullSlice);

    /**
     * Returns a subarray at the given index.
     *
     * @param array the array to get from
     * @param index the index to get
     * @return the subarray
     */
    public NDArray get(NDArray array, NDIndex index) {
        if (index.getRank() == 0 && array.getShape().isScalar()) {
            return array.duplicate();
        }

        // use booleanMask for NDIndexBooleans case
        List<NDIndexElement> indices = index.getIndices();
        if (!indices.isEmpty() && indices.get(0) instanceof NDIndexBooleans) {
            if (indices.size() != 1) {
                throw new IllegalArgumentException(
                        "get() currently didn't support more that one boolean NDArray");
            }
            return array.booleanMask(((NDIndexBooleans) indices.get(0)).getIndex());
        }

        Optional<NDIndexFullPick> fullPick = NDIndexFullPick.fromIndex(index, array.getShape());
        if (fullPick.isPresent()) {
            return get(array, fullPick.get());
        }

        Optional<NDIndexFullSlice> fullSlice = NDIndexFullSlice.fromIndex(index, array.getShape());
        if (fullSlice.isPresent()) {
            return get(array, fullSlice.get());
        }
        throw new UnsupportedOperationException(
                "get() currently supports all, fixed, and slices indices");
    }

    /**
     * Sets the values of the array at the fullSlice with an array.
     *
     * @param array the array to set
     * @param fullSlice the fullSlice of the index to set in the array
     * @param value the value to set with
     */
    public abstract void set(NDArray array, NDIndexFullSlice fullSlice, NDArray value);

    /**
     * Sets the values of the array at the boolean locations with an array.
     *
     * @param array the array to set
     * @param indices a boolean array where true indicates values to update
     * @param value the value to set with when condition is true
     */
    public void set(NDArray array, NDIndexBooleans indices, NDArray value) {
        array.intern(NDArrays.where(indices.getIndex(), value, array));
    }

    /**
     * Sets the values of the array at the index locations with an array.
     *
     * @param array the array to set
     * @param index the index to set at in the array
     * @param value the value to set with
     */
    public void set(NDArray array, NDIndex index, NDArray value) {
        List<NDIndexElement> indices = index.getIndices();
        if (!indices.isEmpty() && indices.get(0) instanceof NDIndexBooleans) {
            if (indices.size() != 1) {
                throw new IllegalArgumentException(
                        "get() currently didn't support more that one boolean NDArray");
            }
            set(array, (NDIndexBooleans) indices.get(0), value);
        }

        NDIndexFullSlice fullSlice =
                NDIndexFullSlice.fromIndex(index, array.getShape()).orElse(null);
        if (fullSlice != null) {
            set(array, fullSlice, value);
            return;
        }
        throw new UnsupportedOperationException(
                "set() currently supports all, fixed, and slices indices");
    }

    /**
     * Sets the values of the array at the fullSlice with a number.
     *
     * @param array the array to set
     * @param fullSlice the fullSlice of the index to set in the array
     * @param value the value to set with
     */
    public abstract void set(NDArray array, NDIndexFullSlice fullSlice, Number value);

    /**
     * Sets the values of the array at the index locations with a number.
     *
     * @param array the array to set
     * @param index the index to set at in the array
     * @param value the value to set with
     */
    public void set(NDArray array, NDIndex index, Number value) {
        NDIndexFullSlice fullSlice =
                NDIndexFullSlice.fromIndex(index, array.getShape()).orElse(null);
        if (fullSlice != null) {
            set(array, fullSlice, value);
            return;
        }
        // use booleanMask for NDIndexBooleans case
        List<NDIndexElement> indices = index.getIndices();
        if (!indices.isEmpty() && indices.get(0) instanceof NDIndexBooleans) {
            if (indices.size() != 1) {
                throw new IllegalArgumentException(
                        "set() currently didn't support more that one boolean NDArray");
            }
            set(array, (NDIndexBooleans) indices.get(0), array.getManager().create(value));
            return;
        }
        throw new UnsupportedOperationException(
                "set() currently supports all, fixed, and slices indices");
    }

    /**
     * Sets a scalar value in the array at the indexed location.
     *
     * @param array the array to set
     * @param index the index to set at in the array
     * @param value the value to set with
     * @throws IllegalArgumentException if the index does not point to a scalar value in the array
     */
    public void setScalar(NDArray array, NDIndex index, Number value) {
        NDIndexFullSlice fullSlice =
                NDIndexFullSlice.fromIndex(index, array.getShape()).orElse(null);
        if (fullSlice != null) {
            if (fullSlice.getShape().size() != 1) {
                throw new IllegalArgumentException("The provided index does not set a scalar");
            }
            set(array, index, value);
            return;
        }
        throw new UnsupportedOperationException(
                "set() currently supports all, fixed, and slices indices");
    }
}
