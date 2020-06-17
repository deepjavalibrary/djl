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
 * {@code MaskedSoftmaxCrossEntropyLoss} is an implementation of {@link Loss} that only considers a
 * specific number of values for the loss computations, and masks the rest according to the given
 * sequence.
 */
public class MaskedSoftmaxCrossEntropyLoss extends Loss {
    private float weight;
    private int classAxis;
    private boolean sparseLabel;
    private boolean fromLogit;

    /** Creates a new instance of {@code SoftmaxCrossEntropyLoss} with default parameters. */
    public MaskedSoftmaxCrossEntropyLoss() {
        this("MaskedSoftmaxCrossEntropyLoss");
    }

    /**
     * Creates a new instance of {@code SoftmaxCrossEntropyLoss} with default parameters.
     *
     * @param name the name of the loss
     */
    public MaskedSoftmaxCrossEntropyLoss(String name) {
        this(name, 1, -1, true, false);
    }

    /**
     * Creates a new instance of {@code MaskedSoftmaxCrossEntropyLoss} with the given parameters.
     *
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether predictions are log probabilities or un-normalized numbers, default
     *     false
     */
    public MaskedSoftmaxCrossEntropyLoss(
            String name, float weight, int classAxis, boolean sparseLabel, boolean fromLogit) {
        super(name);
        this.weight = weight;
        this.classAxis = classAxis;
        this.sparseLabel = sparseLabel;
        this.fromLogit = fromLogit;
    }

    /**
     * Calculates the evaluation between the labels and the predictions. The {@code label} parameter
     * is an {@link NDList} that contains the label and the mask sequence in that order.
     *
     * @param labels the {@link NDList} that contains correct values and the mask sequence
     * @param predictions the predicted values
     * @return the evaluation result
     */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray weights = labels.head().onesLike().expandDims(-1).sequenceMask(labels.get(1));
        NDArray pred = predictions.singletonOrThrow();
        if (!fromLogit) {
            pred = pred.logSoftmax(classAxis);
        }
        NDArray loss;
        NDArray lab = labels.head();
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
        loss = loss.mul(weights);
        if (weight != 1) {
            loss = loss.mul(weight);
        }
        return loss.mean(new int[] {1});
    }
}
