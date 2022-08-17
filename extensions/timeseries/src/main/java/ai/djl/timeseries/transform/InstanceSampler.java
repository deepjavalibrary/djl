/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.transform;

import ai.djl.ndarray.NDArray;

import java.util.List;

/**
 * An InstanceSampler is called with the time series ``ts``, and returns a set of indices at which
 * training instances will be generated.
 *
 * <p>The sampled indices ``i`` satisfy ``a &lt;= i &lt;= b``, where ``a = min_past`` and ``b =
 * ts.shape[axis] - min_future``.
 */
public abstract class InstanceSampler {

    protected int axis;
    protected int minPast;
    protected int minFuture;

    /**
     * Constructs a new instance of {@code InstanceSampler}.
     *
     * @param axis the axis of the time series length
     * @param minPast minimal pastime length
     * @param minFuture minimal future time length
     */
    public InstanceSampler(int axis, int minPast, int minFuture) {
        this.axis = axis;
        this.minPast = minPast;
        this.minFuture = minFuture;
    }

    /**
     * Returns the sampled indices bounds.
     *
     * @param ts the time series
     * @return the indices bound
     */
    public int[] getBounds(NDArray ts) {
        int start = this.minPast;
        int posAxis = this.axis < 0 ? ts.getShape().dimension() + this.axis : this.axis;
        int end = (int) ts.getShape().get(posAxis) - this.minFuture;
        return new int[] {start, end};
    }

    /**
     * Call the sample process.
     *
     * @param ts the time series
     * @return list of indices
     */
    public abstract List<Integer> call(NDArray ts);
}
