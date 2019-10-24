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
import ai.djl.ndarray.NDList;

/**
 * {@code L1Loss} calculates L1 loss between label and prediction.
 *
 * <p>L1 loss is defined by \(L = \sum_i \vert {label}_i - {prediction}_i \vert\).
 */
public class L1Loss extends Loss {

    private float weight;
    private int batchAxis;

    /**
     * Calculates L1 Loss between the label and prediction, a.k.a. MAE(Mean Absolute Error).
     *
     * @param weight the weight to apply on loss value, default 1
     * @param batchAxis the axis that represents mini-batch, default 0
     */
    public L1Loss(float weight, int batchAxis) {
        super("L1Loss");
        this.weight = weight;
        this.batchAxis = batchAxis;
    }

    /** Calculates L1 Loss between the label and prediction, a.k.a. MAE(Mean Absolute Error). */
    public L1Loss() {
        this(1, 0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDList label, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        NDArray labelReshaped = label.singletonOrThrow().reshape(pred.getShape());
        NDArray loss = labelReshaped.sub(pred).abs();
        if (weight != 1) {
            // avoid broadcast mul
            loss = labelReshaped.mul(weight);
        }
        // apply mean on all axes except the batchAxis
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }
}
