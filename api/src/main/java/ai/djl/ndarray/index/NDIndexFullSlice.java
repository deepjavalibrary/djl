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
package ai.djl.ndarray.index;

import ai.djl.ndarray.types.Shape;
import java.util.List;

/**
 * An index as a slice on all dimensions where some dimensions can be squeezed.
 *
 * <p>Create using {@link NDIndex#getAsFullSlice(Shape)}.
 */
public class NDIndexFullSlice {
    private long[] min;
    private long[] max;
    private long[] step;
    private List<Integer> toSqueeze;
    private Shape shape;
    private Shape squeezedShape;

    NDIndexFullSlice(
            long[] min,
            long[] max,
            long[] step,
            List<Integer> toSqueeze,
            Shape shape,
            Shape squeezedShape) {
        this.min = min;
        this.max = max;
        this.step = step;
        this.toSqueeze = toSqueeze;
        this.shape = shape;
        this.squeezedShape = squeezedShape;
    }

    /**
     * Returns the slice min for each axis.
     *
     * @return the slice min for each axis
     */
    public long[] getMin() {
        return min;
    }

    /**
     * Returns the slice max for each axis.
     *
     * @return the slice max for each axis
     */
    public long[] getMax() {
        return max;
    }

    /**
     * Returns the slice step for each axis.
     *
     * @return the slice step for each axis.
     */
    public long[] getStep() {
        return step;
    }

    /**
     * Returns the list of axis to squeeze.
     *
     * @return the list of axis to squeeze
     */
    public List<Integer> getToSqueeze() {
        return toSqueeze;
    }

    /**
     * Returns the slice shape without squeezing.
     *
     * @return the slice shape without squeezing
     */
    public Shape getShape() {
        return shape;
    }

    /**
     * Returns the slice shape with squeezing.
     *
     * @return the slice shape with squeezing
     */
    public Shape getSqueezedShape() {
        return squeezedShape;
    }
}
