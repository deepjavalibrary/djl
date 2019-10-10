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

package ai.djl.training.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;

/** Helper metric class to record loss value and calculate average loss value. */
public class LossMetric extends TrainingMetrics {
    private float totalLoss;
    private int totalInstances;

    public LossMetric(String name) {
        super(name);
    }

    /** {@inheritDoc} */
    @Override
    protected void update(NDList labels, NDList predictions) {
        throw new UnsupportedOperationException(
                "LossMetric does not support update "
                        + "based labels and predictions, it only accepts NDArray loss values");
    }

    /** {@inheritDoc} */
    @Override
    protected void update(NDArray labels, NDArray predictions) {
        throw new UnsupportedOperationException(
                "LossMetric does not support update "
                        + "based labels and predictions, it only accepts NDArray loss values");
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDArray loss) {
        totalLoss += loss.sum().getFloat();
        totalInstances += loss.size();
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDList loss) {
        for (NDArray array : loss.toArray()) {
            totalLoss += array.sum().getFloat();
            totalInstances += array.size();
        }
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        totalLoss = 0.f;
        totalInstances = 0;
    }

    @Override
    public Pair<String, Float> getMetric() {
        if (totalInstances == 0) {
            return new Pair<>(getName(), Float.NaN);
        }
        return new Pair<>(getName(), totalLoss / totalInstances);
    }
}
