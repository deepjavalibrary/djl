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
import ai.djl.ndarray.index.NDIndex;

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
    private int classAxis;
    private boolean sparseLabel;
    private boolean fromLogit;

    /** Creates a new instance of {@code SoftmaxCrossEntropyLoss} with default parameters. */
    public SoftmaxCrossEntropyLoss() {
        this("SoftmaxCrossEntropyLoss");
    }

    /**
     * Creates a new instance of {@code SoftmaxCrossEntropyLoss} with default parameters.
     *
     * @param name the name of the loss
     */
    public SoftmaxCrossEntropyLoss(String name) {
        this(name, 1, -1, true, true);
    }

    /**
     * Creates a new instance of {@code SoftmaxCrossEntropyLoss} with the given parameters.
     *
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are 1-D integer array or 2-D probabilities of [batch_size,
     *     n-class], default true
     * @param fromLogit whether predictions are un-normalized numbers or log probabilities, if true,
     *     logSoftmax will be applied to input, default true
     */
    public SoftmaxCrossEntropyLoss(
            String name, float weight, int classAxis, boolean sparseLabel, boolean fromLogit) {
        super(name);
        this.weight = weight;
        this.classAxis = classAxis;
        this.sparseLabel = sparseLabel;
        this.fromLogit = fromLogit;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList label, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        if (fromLogit) {
            pred = pred.logSoftmax(classAxis);
        }
        NDArray loss;
        NDArray lab = label.singletonOrThrow();
        if (sparseLabel) {
            NDIndex pickIndex =
                    new NDIndex()
                            .addAllDim(Math.floorMod(classAxis, pred.getShape().dimension()))
                            .addPickDim(lab);
            loss = pred.get(pickIndex).neg();
        } else {
            lab = lab.reshape(pred.getShape());
            loss = pred.mul(lab).neg().sum(new int[] {classAxis}, true);
        }
        if (weight != 1) {
            loss = loss.mul(weight);
        }
        return loss.mean();
    }
}
