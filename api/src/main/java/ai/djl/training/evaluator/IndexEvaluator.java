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
package ai.djl.training.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/**
 * A wrapper for an {@link Evaluator} that evaluates on only a particular {@link NDArray} in the
 * predictions and/or labels {@link NDList}s.
 */
public class IndexEvaluator extends Evaluator {

    private Evaluator evaluator;
    private Integer predictionsIndex;
    private Integer labelsIndex;

    /**
     * Constructs an {@link IndexEvaluator} with the same index for both predictions and labels.
     *
     * @param evaluator the base evaluator
     * @param index the index for both predictions and labels
     */
    public IndexEvaluator(Evaluator evaluator, int index) {
        this(evaluator, index, index);
    }

    /**
     * Constructs an {@link IndexEvaluator}.
     *
     * @param evaluator the base evaluator
     * @param predictionsIndex the predictions index
     * @param labelsIndex the labels index
     */
    public IndexEvaluator(Evaluator evaluator, Integer predictionsIndex, Integer labelsIndex) {
        super(evaluator.getName());
        this.evaluator = evaluator;
        this.predictionsIndex = predictionsIndex;
        this.labelsIndex = labelsIndex;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        return evaluator.evaluate(getLabels(labels), getPredictions(predictions));
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        evaluator.addAccumulator(key);
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        evaluator.updateAccumulator(key, getLabels(labels), getPredictions(predictions));
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulators(String[] keys, NDList labels, NDList predictions) {
        evaluator.updateAccumulators(keys, getLabels(labels), getPredictions(predictions));
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        evaluator.resetAccumulator(key);
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        return evaluator.getAccumulator(key);
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
