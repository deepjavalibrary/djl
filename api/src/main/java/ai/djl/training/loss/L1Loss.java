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

public class L1Loss extends Loss {

    private float weight;
    private int batchAxis;

    /**
     * Calculate L1 Loss between label and prediction, a.k.a. MAE(Mean Absolute Error).
     *
     * <p>.. math:: L = \sum_i \vert {label}_i - {prediction}_i \vert.
     *
     * @param weight weight to apply on loss value, default 1
     * @param batchAxis axis that represents mini-batch, default 0
     */
    public L1Loss(float weight, int batchAxis) {
        this.weight = weight;
        this.batchAxis = batchAxis;
    }

    /**
     * Calculate L1 Loss between label and prediction, a.k.a. MAE(Mean Absolute Error).
     *
     * <p>.. math:: L = \sum_i \vert {label}_i - {prediction}_i \vert.
     */
    public L1Loss() {
        weight = 1;
        batchAxis = 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDArray label, NDArray prediction) {
        label = label.reshape(prediction.getShape());
        NDArray loss = label.sub(prediction).abs();
        if (weight != 1) {
            // avoid broadcast mul
            loss = label.mul(weight);
        }
        // apply mean on all axes except the batchAxis
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }
}
