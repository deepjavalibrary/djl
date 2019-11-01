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
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;

/** Calculates L2Loss between label and prediction, a.k.a. MSE(Mean Square Error). */
public class L2Loss extends Loss {

    private float weight;
    private int batchAxis;

    /**
     * Calculates L2Loss between the label and prediction, a.k.a. MSE(Mean Square Error).
     *
     * <p>.. math:: L = \frac{1}{2} \sum_i \vert {label}_i - {prediction}_i \vert^2.
     *
     * @param weight the weight to apply on loss value, default 1/2
     * @param batchAxis the axis that represents mini-batch, default 0
     */
    public L2Loss(float weight, int batchAxis) {
        this.weight = weight;
        this.batchAxis = batchAxis;
    }

    /**
     * Calculate L2Loss between the label and prediction, a.k.a. MSE(Mean Square Error).
     *
     * <p>.. math:: L = \frac{1}{2} \sum_i \vert {label}_i - {prediction}_i \vert^2.
     */
    public L2Loss() {
        weight = 1.f / 2;
        batchAxis = 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDArray label, NDArray prediction) {
        label = label.reshape(prediction.getShape());
        NDArray loss = label.sub(prediction).square().mul(weight);
        // apply mean on all axes except the batchAxis
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }
}
