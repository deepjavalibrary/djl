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
import ai.djl.training.metrics.TrainingMetric;
import java.util.stream.IntStream;

/** Loss functions or Cost Functions to evaluate model predictions against true labels. */
public abstract class Loss extends TrainingMetric {

    private float totalLoss;
    private int totalInstances;

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name The display name of the Loss
     */
    public Loss(String name) {
        super(name);
    }

    /**
     * Calculates loss between the label and prediction.
     *
     * @param label the true label
     * @param prediction the predicted label
     * @return the loss value
     */
    public abstract NDArray getLoss(NDList label, NDList prediction);

    /**
     * Returns a new instance of {@link L1Loss} with default weight and batch axis.
     *
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss() {
        return new L1Loss();
    }

    /**
     * Returns a new instance of {@link L1Loss} with given weight and batch axis.
     *
     * @param weight the weight to apply on loss value, default 1
     * @param batchAxis the axis that represents mini-batch, default 0
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss(float weight, int batchAxis) {
        return new L1Loss(weight, batchAxis);
    }

    /**
     * Returns a new instance of {@link L2Loss} with default weight and batch axis.
     *
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss() {
        return new L2Loss();
    }

    /**
     * Returns a new instance of {@link L2Loss} with given weight and batch axis.
     *
     * @param weight the weight to apply on loss value, default 1
     * @param batchAxis the axis that represents mini-batch, default 0
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss(float weight, int batchAxis) {
        return new L2Loss(weight, batchAxis);
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with default arguments.
     *
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss() {
        return new SigmoidBinaryCrossEntropyLoss();
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with the given arguments.
     *
     * @param weight the weight to apply on the loss value, default 1
     * @param batchAxis the axis that represents the mini-batch, default 0
     * @param fromSigmoid whether the input is from the output of sigmoid, default false
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(
            float weight, int batchAxis, boolean fromSigmoid) {
        return new SigmoidBinaryCrossEntropyLoss(weight, batchAxis, fromSigmoid);
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with default arguments.
     *
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss() {
        return new SoftmaxCrossEntropyLoss();
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with the given arguments.
     *
     * @param weight the weight to apply on the loss value, default 1
     * @param batchAxis the axis that represents the mini-batch, default 0
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(
            float weight, int batchAxis, int classAxis, boolean sparseLabel, boolean fromLogit) {
        return new SoftmaxCrossEntropyLoss(weight, batchAxis, classAxis, sparseLabel, fromLogit);
    }

    /**
     * Returns a new instance of {@link HingeLoss} with default arguments.
     *
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss() {
        return new HingeLoss();
    }

    /**
     * Returns a new instance of {@link HingeLoss} with the given arguments.
     *
     * @param margin the margin in hinge loss. Defaults to 1.0
     * @param weight the weight to apply on loss value, default 1
     * @param batchAxis the axis that represents mini-batch, default 0
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss(int margin, float weight, int batchAxis) {
        return new HingeLoss(margin, weight, batchAxis);
    }

    /** {@inheritDoc} */
    @Override
    public Loss duplicate() {
        try {
            return (Loss) clone();
        } catch (CloneNotSupportedException e) {
            // ignore
            throw new AssertionError("Clone is not supported", e);
        }
    }

    /**
     * Updates the training metrics based on a {@link NDList} of labels and predictions.
     *
     * <p>This is a synchronized operation. You should only call it at the end of a batch or epoch.
     *
     * @param labels a {@code NDList} of labels
     * @param predictions a {@code NDList} of predictions
     */
    @Override
    public void update(NDList labels, NDList predictions) {
        // this is a synchronized operation, only call it at end of batch or epoch
        NDArray update = getLoss(labels, predictions);
        totalLoss += update.sum().getFloat();
        totalInstances += update.size();
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        totalLoss = 0.f;
        totalInstances = 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getValue() {
        if (totalInstances == 0) {
            return Float.NaN;
        }
        return totalLoss / totalInstances;
    }

    /**
     * Gets all axes except the batch axis because loss functions require reduction on all axes
     * except the batch axis.
     *
     * @param loss the loss {@code NDArray}
     * @param batchAxis the axis that represents the mini-batch
     * @return all axes except the batch axis
     */
    protected int[] excludeBatchAxis(NDArray loss, int batchAxis) {
        return IntStream.range(0, loss.getShape().dimension())
                .filter(axis -> axis != batchAxis)
                .toArray();
    }
}
