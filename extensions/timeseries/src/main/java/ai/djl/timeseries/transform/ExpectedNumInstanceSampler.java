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
import java.util.Random;

/**
 * Keeps track of the average time series length and adjusts the probability per time point such
 * that on average `num_instances` training examples are generated per time series.
 */
public class ExpectedNumInstanceSampler extends InstanceSampler {

    private double numInstances;
    private int totalLength;
    private int n;

    /**
     * Construct a new instance of {@code ExpectedNumInstanceSampler}.
     *
     * @param axis the axis of the time series length
     * @param minPast minimal pastime length
     * @param minFuture minimal future time length
     * @param numInstances number of training examples generated per time series on average
     */
    public ExpectedNumInstanceSampler(int axis, int minPast, int minFuture, double numInstances) {
        super(axis, minPast, minFuture);
        this.numInstances = numInstances;
    }

    /** {@inheritDoc} */
    @Override
    public List<Integer> call(NDArray ts) {
        int[] bound = getBounds(ts);
        int windowSize = bound[1] - bound[0] + 1;

        if (windowSize <= 0) {
            return new ArrayList<>();
        }

        n += 1;
        totalLength += windowSize;
        int avgLength = totalLength / n;

        if (avgLength <= 0) {
            return new ArrayList<>();
        }

        double prob = numInstances / avgLength;
        List<Integer> indices = new ArrayList<>();
        Random random = new Random();
        while (indices.isEmpty()) {
            for (int i = 0; i < windowSize; i++) {
                if (random.nextDouble() < prob) {
                    indices.add(i + bound[0]);
                }
            }
        }

        return indices;
    }
}
