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

package software.amazon.ai.training.metrics;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.util.Pair;

/**
 * Computes accuracy classification score.
 *
 * <p>The accuracy score is defined as .. math:: \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n}
 * \\sum_{i=0}^{n-1} \\text{1}(\\hat{y_i} == y_i)
 */
public class Accuracy extends TrainingMetrics {

    private int correctInstances;
    private int totalInstances;
    protected int axis;

    /**
     * Creates Accuracy metric.
     *
     * @param name name of the metric, default is "Accuracy"
     * @param axis axis the axis that represent classes in prediction, default 1
     */
    public Accuracy(String name, int axis) {
        super(name);
        this.axis = axis;
    }

    public Accuracy() {
        this("Accuracy", 1);
    }

    public Accuracy(String name) {
        this(name, 1);
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        correctInstances = 0;
        totalInstances = 0;
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDArray labels, NDArray predictions) {
        checkLabelShapes(labels, predictions);
        NDArray predictionReduced;
        if (labels.getShape() != predictions.getShape()) {
            predictionReduced = predictions.argmax(axis, false);
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
        // Accuracy only support one type of label/prediction
        if (labels.size() != 1 || predictions.size() != 1) {
            throw new IllegalArgumentException(
                    "NDList labels and prediction size "
                            + "must be 1 for Accuracy. For batch data please use a NDArray with first"
                            + "dimension as batch axis.");
        }
        update(labels.get(0), predictions.get(0));
    }

    /** {@inheritDoc} */
    @Override
    protected void update(NDArray loss) {
        throw new UnsupportedOperationException(
                "Accuracy does not support update based on loss NDArray.");
    }

    /** {@inheritDoc} */
    @Override
    public Pair<String, Float> getMetric() {
        if (totalInstances == 0) {
            return new Pair<>(getName(), Float.NaN);
        }
        return new Pair<>(getName(), (float) correctInstances / totalInstances);
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
