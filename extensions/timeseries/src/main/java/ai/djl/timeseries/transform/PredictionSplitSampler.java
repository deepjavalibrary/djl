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

import java.util.ArrayList;
import java.util.List;

/** Sampler used for prediction. */
public class PredictionSplitSampler extends InstanceSampler {

    private boolean allowEmptyInterval;

    /**
     * Constructor for {@link PredictionSplitSampler}.
     *
     * @param axis the axis of the time series length
     * @param minPast minimal pastime length
     * @param minFuture minimal future time length
     * @param allowEmptyInterval whether allow to output an empty {@link NDArray}
     */
    public PredictionSplitSampler(
            int axis, int minPast, int minFuture, boolean allowEmptyInterval) {
        super(axis, minPast, minFuture);
        this.allowEmptyInterval = allowEmptyInterval;
    }

    /** {@inheritDoc} * */
    @Override
    public List<Integer> call(NDArray ts) {
        int[] bound = this.getBounds(ts);
        List<Integer> ret = new ArrayList<>();
        if (bound[0] < bound[1]) {
            ret.add(bound[1]);
        } else if (!allowEmptyInterval) {
            throw new IllegalArgumentException("The start >= end while allowEmptyInterval = False");
        }
        return ret;
    }

    /**
     * Constructs a SplitSampler for test.
     *
     * @return a {@link PredictionSplitSampler}
     */
    public static PredictionSplitSampler newTestSplitSampler() {
        return new PredictionSplitSampler(-1, 0, 0, false);
    }

    /**
     * Constructs a SplitSampler for valid.
     *
     * @return a {@link PredictionSplitSampler}
     */
    public static PredictionSplitSampler newValidationSplitSampler() {
        return new PredictionSplitSampler(-1, 0, 0, true);
    }
}
