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
 * {@code SoftmaxCrossEntropyLoss} is a type of {@link Loss} that calculates the softmax cross
 * entropy loss.
 *
 * <p>If {@code sparse_label} is {@code true} (default), {@code label} should contain integer
 * category indicators. Then, \(L = -\sum_i \log p_{i, label_i}\). If {@code sparse_label} is {@code
 * false}, {@code label} should contain probability distribution and its shape should be the same as
 * the shape of {@code prediction}. Then, \(L = -\sum_i \sum_j {label}_j \log p_{ij}\).
 */
public class SoftmaxCrossEntropyLoss extends Loss {

    private float weight;
    private int batchAxis;
    private int classAxis;
    private boolean sparseLabel;
    private boolean fromLogit;

    /**
     * Creates a new instance of {@code SoftmaxCrossEntropyLoss} with the given parameters.
     *
     * @param weight the weight to apply on the loss value, default 1
     * @param batchAxis the axis that represents the mini-batch, default 0
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     */
    public SoftmaxCrossEntropyLoss(
            float weight, int batchAxis, int classAxis, boolean sparseLabel, boolean fromLogit) {
        super("SoftmaxCrossEntropyLoss");
        this.weight = weight;
        this.batchAxis = batchAxis;
        this.classAxis = classAxis;
        this.sparseLabel = sparseLabel;
        this.fromLogit = fromLogit;
    }

    /** Creates a new instance of {@code SoftmaxCrossEntropyLoss} with default parameters. */
    public SoftmaxCrossEntropyLoss() {
        this(1, 0, -1, true, false);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDList label, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        if (!fromLogit) {
            // TODO: use numpy log softmax
            pred = pred.softmax(classAxis).log();
        }
        NDArray loss;
        NDArray lab = label.singletonOrThrow();
        if (sparseLabel) {
            loss = pred.getNDArrayInternal().pick(lab, classAxis, true).neg();
        } else {
            lab = lab.reshape(pred.getShape());
            loss = pred.mul(lab).sum(new int[] {classAxis});
        }
        if (weight != 1) {
            loss = loss.mul(weight);
        }
        // apply mean on all axes except the batchAxis
        // TODO: alternative use numpy batch flatten and mean batch axis
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }
}
