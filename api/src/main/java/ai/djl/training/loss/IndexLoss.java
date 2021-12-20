/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
 * A wrapper for a {@link Loss} that evaluates on only a particular {@link NDArray} in the
 * predictions and/or labels {@link NDList}s.
 */
public class IndexLoss extends Loss {

    private Loss loss;
    private Integer predictionsIndex;
    private Integer labelsIndex;

    /**
     * Constructs an {@link IndexLoss} with the same index for both predictions and labels.
     *
     * @param loss the base evaluator
     * @param index the index for both predictions and labels
     */
    public IndexLoss(Loss loss, int index) {
        this(loss, index, index);
    }

    /**
     * Constructs an {@link IndexLoss}.
     *
     * @param loss the base evaluator
     * @param predictionsIndex the predictions index
     * @param labelsIndex the labels index
     */
    public IndexLoss(Loss loss, Integer predictionsIndex, Integer labelsIndex) {
        super(loss.getName());
        this.loss = loss;
        this.predictionsIndex = predictionsIndex;
        this.labelsIndex = labelsIndex;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        return loss.evaluate(getLabels(labels), getPredictions(predictions));
    }

    private NDList getPredictions(NDList predictions) {
        if (predictionsIndex == null) {
            return predictions;
        }
        return new NDList(predictions.get(predictionsIndex));
    }

    private NDList getLabels(NDList labels) {
        if (labelsIndex == null) {
            return labels;
        }
        return new NDList(labels.get(labelsIndex));
    }
}
