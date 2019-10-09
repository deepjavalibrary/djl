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
package software.amazon.ai.training.loss;

import software.amazon.ai.ndarray.NDArray;

public class SoftmaxCrossEntropyLoss extends Loss {

    private float weight;
    private int batchAxis;
    private int classAxis;
    private boolean sparseLabel;
    private boolean fromLogit;

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
     * @param weight weight to apply on loss value, default 1
     * @param batchAxis axis that represents mini-batch, default 0
     * @param classAxis axis that represents class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     */
    public SoftmaxCrossEntropyLoss(
            float weight, int batchAxis, int classAxis, boolean sparseLabel, boolean fromLogit) {
        this.weight = weight;
        this.batchAxis = batchAxis;
        this.classAxis = classAxis;
        this.sparseLabel = sparseLabel;
        this.fromLogit = fromLogit;
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
     */
    public SoftmaxCrossEntropyLoss() {
        weight = 1;
        batchAxis = 0;
        classAxis = -1;
        sparseLabel = true;
        fromLogit = false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDArray label, NDArray prediction) {
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
}
