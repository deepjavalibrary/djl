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
package software.amazon.ai.ndarray.index;

import java.util.List;
import software.amazon.ai.ndarray.types.Shape;

public class NDIndexFullSlice {
    private long[] min;
    private long[] max;
    private long[] step;
    private List<Integer> toSqueeze;
    private Shape shape;
    private Shape squeezedShape;

    public NDIndexFullSlice(
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

    public long[] getMin() {
        return min;
    }

    public long[] getMax() {
        return max;
    }

    public long[] getStep() {
        return step;
    }

    public List<Integer> getToSqueeze() {
        return toSqueeze;
    }

    public Shape getShape() {
        return shape;
    }

    public Shape getSqueezedShape() {
        return squeezedShape;
    }
}
