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
package software.amazon.ai.training;

import java.util.stream.IntStream;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;

/** Loss functions or Cost Functions to evaluate model predictions against true labels. */
public final class Loss {

    private Loss() {}

    /**
     * Calculate L2Loss between label and prediction, a.k.a. MSE(Mean Square Error).
     *
     * <p>.. math:: L = \frac{1}{2} \sum_i \vert {label}_i - {prediction}_i \vert^2.
     *
     * @param label true label
     * @param prediction predicted label
     * @param weight weight to apply on loss value, default 1/2
     * @param batchAxis axis that represents mini-batch, default 0
     * @return L2 loss value
     */
    public static NDArray l2Loss(NDArray label, NDArray prediction, float weight, int batchAxis) {
        label = label.reshape(prediction.getShape());
        NDArray loss = label.sub(prediction).square().mul(weight);
        // apply mean on all axes except the batchAxis
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }

    public static NDArray l2Loss(NDArray label, NDArray prediction) {
        return l2Loss(label, prediction, 1.f / 2, 0);
    }

    /**
     * Calculate L1Loss between label and prediction, a.k.a. MAE(Mean Absolute Error).
     *
     * <p>.. math:: L = \sum_i \vert {label}_i - {prediction}_i \vert.
     *
     * @param label true label
     * @param prediction predicted label
     * @param weight weight to apply on loss value, default 1
     * @param batchAxis axis that represents mini-batch, default 0
     * @return L1 loss value
     */
    public static NDArray l1Loss(NDArray label, NDArray prediction, float weight, int batchAxis) {
        label = label.reshape(prediction.getShape());
        NDArray loss = label.sub(prediction).abs();
        if (weight != 1) {
            // avoid broadcast mul
            loss = label.mul(weight);
        }
        // apply mean on all axes except the batchAxis
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }

    public static NDArray l1Loss(NDArray label, NDArray prediction) {
        return l1Loss(label, prediction, 1, 0);
    }

    /**
     * The Sigmoid cross-entropy loss for binary classification.
     *
     * @param label true label
     * @param prediction predicted label
     * @param weight weight to apply on loss value, default 1
     * @param batchAxis axis that represents mini-batch, default 0
     * @param fromSigmoid Whether the input is from the output of sigmoid, default false
     * @return sigmoid cross-entropy loss value
     */
    public static NDArray sigmoidBinaryCrossEntropyLoss(
            NDArray label, NDArray prediction, float weight, int batchAxis, boolean fromSigmoid) {
        label = label.reshape(prediction.getShape());
        NDArray loss;
        if (!fromSigmoid) {
            // TODO: Add Position weight option
            loss =
                    Activation.relu(prediction)
                            .sub(prediction.mul(label))
                            .add(Activation.softrelu(prediction.abs().neg()));
        } else {
            double eps = 1e-12;
            loss =
                    prediction
                            .add(eps)
                            .log()
                            .mul(label)
                            .add(
                                    NDArrays.sub(1., prediction)
                                            .add(eps)
                                            .mul(NDArrays.sub(1., label)));
        }
        if (weight != 1f) {
            loss = loss.mul(weight);
        }
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }

    public static NDArray sigmoidBinaryCrossEntropyLoss(NDArray label, NDArray prediction) {
        return sigmoidBinaryCrossEntropyLoss(label, prediction, 1, 0, false);
    }

    /**
     * Calculate the softmax cross entropy loss. If {@code sparse_label} is {@code true} (default),
     * {@code label} should contain integer category indicators:
     *
     * <p>.. math:: \DeclareMathOperator{softmax}{softmax} p = \softmax({prediction}) L = -\sum_i
     * \log p_{i,{label}_i}
     *
     * <p>If {@code sparse_label} is {@code false}, {@code label} should contain probability
     * distribution and {@code label}'s shape should be the same with {@code prediction}:
     *
     * <p>.. math:: p = \softmax({prediction}) L = -\sum_i \sum_j {label}_j \log p_{ij}
     *
     * @param label true label
     * @param prediction predicted label
     * @param weight weight to apply on loss value, default 1
     * @param batchAxis axis that represents mini-batch, default 0
     * @param classAxis axis that represents class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     * @return softmax cross-entropy loss value
     */
    public static NDArray softmaxCrossEntropyLoss(
            NDArray label,
            NDArray prediction,
            float weight,
            int batchAxis,
            int classAxis,
            boolean sparseLabel,
            boolean fromLogit) {

        if (!fromLogit) {
            prediction = prediction.softmax(classAxis).log();
        }
        NDArray loss;
        if (sparseLabel) {
            loss = prediction.getNDArrayInternal().pick(label, classAxis, true).neg();
        } else {
            label = label.reshape(prediction.getShape());
            loss = prediction.mul(label).sum(new int[] {classAxis});
        }
        if (weight != 1) {
            loss = loss.mul(weight);
        }
        // apply mean on all axes except the batchAxis
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }

    public static NDArray softmaxCrossEntropyLoss(NDArray label, NDArray prediction) {
        return softmaxCrossEntropyLoss(label, prediction, 1, 0, -1, true, false);
    }

    /**
     * Calculate Hinge loss.
     *
     * <p>.. math:: L = \sum_i max(0, {margin} - {pred}_i \cdot {label}_i)
     *
     * @param label true label
     * @param prediction predicted label
     * @param margin The margin in hinge loss. Defaults to 1.0
     * @param weight weight to apply on loss value, default 1
     * @param batchAxis axis that represents mini-batch, default 0
     * @return hinge loss value
     */
    public static NDArray hingeLoss(
            NDArray label, NDArray prediction, int margin, float weight, int batchAxis) {
        label = label.reshape(prediction.getShape());
        NDArray loss = Activation.relu(NDArrays.sub(margin, label.mul(prediction)));
        if (weight != 1) {
            loss = loss.mul(weight);
        }
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }

    public static NDArray hingeLoss(NDArray label, NDArray prediction) {
        return hingeLoss(label, prediction, 1, 1, 0);
    }

    /**
     * Helper function to get all axes except batch axis, loss functions requires reduction on all
     * axes except batch axis.
     *
     * @param loss loss {@code NDArray}
     * @param batchAxis axis that represents mini-batch
     * @return all axes except batch axis
     */
    private static int[] excludeBatchAxis(NDArray loss, int batchAxis) {
        return IntStream.range(0, loss.getShape().dimension())
                .filter(axis -> axis != batchAxis)
                .toArray();
    }
}
