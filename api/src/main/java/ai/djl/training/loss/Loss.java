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
import ai.djl.training.metrics.TrainingMetrics;
import ai.djl.util.Pair;
import java.util.stream.IntStream;

/** Loss functions or Cost Functions to evaluate model predictions against true labels. */
public abstract class Loss extends TrainingMetrics {

    private float totalLoss;
    private int totalInstances;

    /** Base class for metric with abstract update methods. */
    public Loss() {
        super("Loss");
    }

    /**
     * Calculate loss between label and prediction.
     *
     * @param label true label
     * @param prediction predicted label
     * @return loss value
     */
    public abstract NDArray getLoss(NDArray label, NDArray prediction);

    /**
     * Calculate loss between label and prediction.
     *
     * <p>the default implementation is simply adding all losses together
     *
     * @param labels true labels
     * @param predictions predicted labels
     * @return loss value
     */
    public NDArray getTotalLoss(NDList labels, NDList predictions) {
        // TODO: add weighted loss, used in batch loss calculation
        NDArray loss = getLoss(labels.head(), predictions.head());
        for (int i = 1; i < labels.size(); i++) {
            loss.add(getLoss(labels.get(i), predictions.get(i)));
        }
        return loss;
    }

    public static L1Loss l1Loss() {
        return new L1Loss();
    }

    public static L1Loss l1Loss(float weight, int batchAxis) {
        return new L1Loss(weight, batchAxis);
    }

    public static L2Loss l2Loss() {
        return new L2Loss();
    }

    public static L2Loss l2Loss(float weight, int batchAxis) {
        return new L2Loss(weight, batchAxis);
    }

    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss() {
        return new SigmoidBinaryCrossEntropyLoss();
    }

    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(
            float weight, int batchAxis, boolean fromSigmoid) {
        return new SigmoidBinaryCrossEntropyLoss(weight, batchAxis, fromSigmoid);
    }

    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss() {
        return new SoftmaxCrossEntropyLoss();
    }

    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(
            float weight, int batchAxis, int classAxis, boolean sparseLabel, boolean fromLogit) {
        return new SoftmaxCrossEntropyLoss(weight, batchAxis, classAxis, sparseLabel, fromLogit);
    }

    public static HingeLoss hingeLoss() {
        return new HingeLoss();
    }

    public static HingeLoss hingeLoss(int margin, float weight, int batchAxis) {
        return new HingeLoss(margin, weight, batchAxis);
    }

    /**
     * Helper function to get all axes except batch axis, loss functions requires reduction on all
     * axes except batch axis.
     *
     * @param loss loss {@code NDArray}
     * @param batchAxis axis that represents mini-batch
     * @return all axes except batch axis
     */
    int[] excludeBatchAxis(NDArray loss, int batchAxis) {
        return IntStream.range(0, loss.getShape().dimension())
                .filter(axis -> axis != batchAxis)
                .toArray();
    }

    @Override
    public Loss duplicate() {
        throw new UnsupportedOperationException("Duplicate is not supported for Loss");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray update(NDList labels, NDList predictions) {
        NDArray loss = getTotalLoss(labels, predictions);
        totalLoss += loss.sum().getFloat();
        totalInstances += loss.size();
        return loss;
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
