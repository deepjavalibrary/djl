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
 * {@code Accuracy} is a {@link TrainingMetric} that computes the accuracy score.
 *
 * <p>The accuracy score is defined as \(accuracy(y, \hat{y}) =
 * \frac{1}{n}\sum_{i=0}^{n-1}1(\hat{y_i} == y_i)\)
 */
public class Accuracy extends TrainingMetric {

    private long correctInstances;
    private long totalInstances;
    protected int axis;
    protected int index;

    /**
     * Creates an accuracy metric.
     *
     * @param name the name of the metric, default is "Accuracy"
     * @param index the index of the NDArray in labels to compute accuracy for
     * @param axis the axis that represent classes in prediction, default 1
     */
    public Accuracy(String name, int index, int axis) {
        super(name);
        this.axis = axis;
        this.index = index;
    }

    /** Creates an accuracy metric that computes accuracy across axis 1 along the 0th index. */
    public Accuracy() {
        this("Accuracy", 0, 1);
    }

    /**
     * Creates an accuracy metric that computes accuracy across axis 1 along given index.
     *
     * @param name the name of the metric, default is "Accuracy"
     * @param index the index of the NDArray in labels to compute accuracy for
     */
    public Accuracy(String name, int index) {
        this(name, index, 1);
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        correctInstances = 0;
        totalInstances = 0;
    }

    /**
     * Computes and updates the accuracy based on the labels and predictions.
     *
     * @param labels a {@link NDList} of labels
     * @param predictions a {@link NDList} of predictions
     */
    public void update(NDArray labels, NDArray predictions) {
        checkLabelShapes(labels, predictions);
        NDArray predictionReduced;
        if (!labels.getShape().equals(predictions.getShape())) {
            predictionReduced = predictions.argMax(axis);
        } else {
            predictionReduced = predictions;
        }
        // result of sum operator is int64 now
        long numCorrect =
                labels.asType(DataType.INT64, false)
                        .eq(predictionReduced.asType(DataType.INT64, false))
                        .countNonzero()
                        .getLong();
        addCorrectInstances(numCorrect);
        addTotalInstances(labels.size());
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
     * Add a number to the correct instances.
     *
     * @param numInstances the number to increment by
     */
    public void addCorrectInstances(long numInstances) {
        this.correctInstances += numInstances;
    }

    /**
     * Add a number to the total instances.
     *
     * @param totalInstances the number to increment by
     */
    public void addTotalInstances(long totalInstances) {
        this.totalInstances += totalInstances;
    }
}
