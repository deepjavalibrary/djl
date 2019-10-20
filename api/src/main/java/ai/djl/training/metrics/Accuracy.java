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

package ai.djl.training.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;

/**
 * Computes accuracy classification score.
 *
 * <p>The accuracy score is defined as .. math:: \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n}
 * \\sum_{i=0}^{n-1} \\text{1}(\\hat{y_i} == y_i)
 */
public class Accuracy extends TrainingMetric {

    private int correctInstances;
    private int totalInstances;
    protected int axis;
    protected int index;

    /**
     * Creates Accuracy metric.
     *
     * @param name name of the metric, default is "Accuracy"
     * @param index index of the NDArray in labels to compute accuracy for
     * @param axis axis the axis that represent classes in prediction, default 1
     */
    public Accuracy(String name, int index, int axis) {
        super(name);
        this.axis = axis;
        this.index = index;
    }

    public Accuracy() {
        this("Accuracy", 0, 1);
    }

    public Accuracy(String name, int index) {
        this(name, index, 1);
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        correctInstances = 0;
        totalInstances = 0;
    }

    public void update(NDArray labels, NDArray predictions) {
        checkLabelShapes(labels, predictions);
        NDArray predictionReduced;
        if (labels.getShape() != predictions.getShape()) {
            predictionReduced = predictions.argmax(axis);
        } else {
            predictionReduced = predictions;
        }
        // TODO: remove asType after bug in numpy argmax is fixed. argmax should return int values.
        int numCorrect =
                labels.asType(DataType.INT32, false)
                        .eq(predictionReduced.asType(DataType.INT32, false))
                        .sum()
                        .getInt();
        addCorrectInstances(numCorrect);
        addTotalInstances((int) labels.size());
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDList labels, NDList predictions) {
        if (labels.size() != predictions.size()) {
            throw new IllegalArgumentException("labels and prediction length does not match.");
        }
        update(labels.get(index), predictions.get(index));
    }

    /** {@inheritDoc} */
    @Override
    public float getValue() {
        if (totalInstances == 0) {
            return Float.NaN;
        }
        return (float) correctInstances / totalInstances;
    }

    /**
     * Add number to correct instances.
     *
     * @param numInstances the number to increment
     */
    public void addCorrectInstances(int numInstances) {
        this.correctInstances += numInstances;
    }

    /**
     * Add number to total instances.
     *
     * @param totalInstances the number to increment
     */
    public void addTotalInstances(int totalInstances) {
        this.totalInstances += totalInstances;
    }
}
