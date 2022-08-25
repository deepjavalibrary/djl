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
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;

/**
 * {@code QuantileL1Loss} calculates the Weighted Quantile Loss between labels and predictions. It
 * is useful for regression problems where you wish to estimate a particular quantile. For example,
 * to target the P90, instantiate {@code new QuantileL1Loss("P90", 0.90)}. Basically, what this loss
 * function does is to focus on a centain persentile of the data. Eg. q=0.5 is the original default
 * case of regression, meaning the best-fit line lies in the center. When q=0.9, the best-fit line
 * will lie above the center; and, if \partial forecast / \partial w are the same, then exactly 0.9
 * of total data points will lie below the best-fit line.
 *
 * <pre>
 *  def quantile_loss(target, forecast, q):
 *      return 2 * np.sum(np.abs((forecast - target) * ((target &lt;= forecast) - q)))
 * </pre>
 *
 * <p>Reference: <a href="https://bibinmjose.github.io/2021/03/08/errorblog.html">...</a>
 */
public class QuantileL1Loss extends Loss {

    private Number quantile;

    /** Computes QuantileL1Loss for regression problem. */
    public QuantileL1Loss(float quantile) {
        this("QuantileL1Loss", quantile);
    }

    /** Computes QuantileL1Loss for regression problem. */
    public QuantileL1Loss(String name, float quantile) {
        super(name);
        this.quantile = quantile;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray pred = predictions.singletonOrThrow();
        NDArray labelReshaped = labels.singletonOrThrow().reshape(pred.getShape());
        NDArray loss =
                pred.sub(labelReshaped)
                        .mul(labelReshaped.lte(pred).toType(DataType.FLOAT32, false).sub(quantile))
                        .abs()
                        .mul(2);
        return loss.mean();
    }
}
