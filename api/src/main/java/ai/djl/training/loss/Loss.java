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

/**
 * Loss functions (or Cost functions) are used to evaluate the model predictions against true labels
 * for optimization.
 *
 * <p>Although all training metrics can be used to measure the performance of a model, not all of
 * them are suited to being used by an optimizer. Loss functions are usually non-negative where a
 * larger loss represents worse performance. They are also real-valued to accurately compare models.
 *
 * <p>When creating a loss function, you should avoid having the loss depend on the batch size. For
 * example, if you have a loss per item in a batch and sum those losses, your loss would be {@code
 * numItemsInBatch*avgLoss}. Instead, you should take the mean of those losses to reduce out the
 * batchSize factor. Otherwise, it can make it difficult to tune the learning rate since any change
 * in the batch size would throw it off. If you have a variable batch size, it would be even more
 * difficult.
 */
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
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss(float weight) {
        return new L1Loss(weight);
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
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss(float weight) {
        return new L2Loss(weight);
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
     * @param fromSigmoid whether the input is from the output of sigmoid, default false
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(
            float weight, boolean fromSigmoid) {
        return new SigmoidBinaryCrossEntropyLoss(weight, fromSigmoid);
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
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(
            float weight, int classAxis, boolean sparseLabel, boolean fromLogit) {
        return new SoftmaxCrossEntropyLoss(weight, classAxis, sparseLabel, fromLogit);
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
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss(int margin, float weight) {
        return new HingeLoss(margin, weight);
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
}
