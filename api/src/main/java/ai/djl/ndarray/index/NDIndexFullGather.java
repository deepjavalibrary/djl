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

/** A simplified representation of a gather-based {@link NDIndex}-like class. */
public final class NDIndexFullGather {

    private NDArray indices;
    private int axis;

    /**
     * Constructs a new {@link NDIndexFullGather}.
     *
     * @param indices the indices to gather
     * @param axis the axis to gather from
     */
    NDIndexFullGather(NDArray indices, int axis) {
        this.indices = indices;
        this.axis = axis;
    }

    /**
     * Returns the indices to gather.
     *
     * @return the indices to gather
     */
    public NDArray getIndices() {
        return indices;
    }

    /**
     * Returns the axis to gather.
     *
     * @return the axis to gather
     */
    public int getAxis() {
        return axis;
    }
}
