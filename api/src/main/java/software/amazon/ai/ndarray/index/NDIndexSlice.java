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

/** An NDIndexElement that returns a range of values in the specified dimension. */
public class NDIndexSlice implements NDIndexElement {

    private Integer min;
    private Integer max;
    private Integer step;

    public NDIndexSlice(Integer min, Integer max, Integer step) {
        this.min = min;
        this.max = max;
        this.step = step;
    }

    public Integer getMin() {
        return min;
    }

    public Integer getMax() {
        return max;
    }

    public Integer getStep() {
        return step;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return 1;
    }
}
