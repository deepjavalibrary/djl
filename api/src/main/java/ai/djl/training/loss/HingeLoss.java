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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;

/**
 * {@code HingeLoss} is a type of {@link Loss}.
 *
 * <p>Hinge loss is defined by: \(L = \sum_i max(0, margin - pred_i \cdot label_i)\)
 */
public class HingeLoss extends Loss {

    private int margin;
    private float weight;
    private int batchAxis;

    /**
     * Calculates Hinge loss.
     *
     * @param margin the margin in hinge loss. Defaults to 1.0
     * @param weight the weight to apply on loss value, default 1
     * @param batchAxis the axis that represents mini-batch, default 0
     */
    public HingeLoss(int margin, float weight, int batchAxis) {
        super("HingeLoss");
        this.margin = margin;
        this.weight = weight;
        this.batchAxis = batchAxis;
    }

    /** Calculates Hinge loss. */
    public HingeLoss() {
        this(1, 1, 0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDList label, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        NDArray labelReshaped = label.singletonOrThrow().reshape(pred.getShape());
        NDArray loss = Activation.relu(NDArrays.sub(margin, labelReshaped.mul(pred)));
        if (weight != 1) {
            loss = loss.mul(weight);
        }
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }
}
