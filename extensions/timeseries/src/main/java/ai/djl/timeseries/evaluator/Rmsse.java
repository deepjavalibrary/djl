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

package ai.djl.timeseries.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.util.Pair;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A class used to calculate Root Mean Squared Scaled Error. */
public class Rmsse extends Evaluator {

    private DistributionOutput distributionOutput;
    private int axis;
    private Map<String, Float> totalLoss;

    /**
     * Creates an evaluator that computes Root Mean Squared Scaled Error across axis 1.
     *
     * <p>Please referring <a
     * href=https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/evaluation>https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/evaluation</a>
     * for more details.
     *
     * @param distributionOutput the {@link DistributionOutput} to construct the target distribution
     */
    public Rmsse(DistributionOutput distributionOutput) {
        this("RMSSE", 1, distributionOutput);
    }

    /**
     * Creates an evaluator that computes Root Mean Squared Scaled Error across axis 1.
     *
     * <p>Please referring <a
     * href=https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/evaluation>https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/evaluation</a>
     * for more details.
     *
     * @param name the name of the evaluator, default is "RMSSE"
     * @param axis the axis that represent time length in prediction, default 1
     * @param distributionOutput the {@link DistributionOutput} to construct the target distribution
     */
    public Rmsse(String name, int axis, DistributionOutput distributionOutput) {
        super(name);
        this.axis = axis;
        this.distributionOutput = distributionOutput;
        totalLoss = new ConcurrentHashMap<>();
    }

    protected Pair<Long, NDArray> evaluateHelper(NDList labels, NDList predictions) {
        NDArray label = labels.head();
        NDArray prediction =
                distributionOutput.distributionBuilder().setDistrArgs(predictions).build().mean();

        checkLabelShapes(label, prediction);
        NDArray meanSquare = label.sub(prediction).square().mean(new int[] {axis});
        NDArray scaleDenom =
                label.get(":, 1:").sub(label.get(":, :-1")).square().mean(new int[] {axis});

        NDArray rmsse = meanSquare.div(scaleDenom).sqrt();
        rmsse = NDArrays.where(scaleDenom.eq(0), rmsse.onesLike(), rmsse);
        long total = rmsse.countNonzero().getLong();

        return new Pair<>(total, rmsse);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        return evaluateHelper(labels, predictions).getValue();
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        totalInstances.put(key, 0L);
        totalLoss.put(key, 0f);
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        Pair<Long, NDArray> update = evaluateHelper(labels, predictions);
        totalInstances.compute(key, (k, v) -> v + update.getKey());
        totalLoss.compute(
                key,
                (k, v) -> {
                    try (NDArray array = update.getValue().sum()) {
                        return v + array.getFloat();
                    }
                });
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        totalInstances.compute(key, (k, v) -> 0L);
        totalLoss.compute(key, (k, v) -> 0f);
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        Long total = totalInstances.get(key);
        if (total == null || total == 0) {
            return Float.NaN;
        }

        return (float) totalLoss.get(key) / totalInstances.get(key);
    }
}
