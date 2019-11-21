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
import ai.djl.util.Pair;

/**
 * Base class for all training metrics.
 *
 * <p>Training metrics can be used to evaluate the performance of a model. They can all be monitored
 * to make an assessment about the performance of the model. However, only ones that further extend
 * {@link ai.djl.training.loss.Loss} are suited to being used to directly optimize a model.
 */
public abstract class TrainingMetric implements Cloneable {

    private String name;

    /**
     * Creates a metric with abstract update methods.
     *
     * @param name the name of the metric
     */
    public TrainingMetric(String name) {
        this.name = name;
    }

    /**
     * Creates and returns a copy of this object.
     *
     * @return a copy of this {@code TrainingMetric}
     */
    public TrainingMetric duplicate() {
        try {
            return (TrainingMetric) clone();
        } catch (CloneNotSupportedException e) {
            // ignore
            throw new AssertionError("Clone is not supported", e);
        }
    }

    /**
     * Computes and updates the training metrics based on the labels and predictions.
     *
     * @param labels a {@code NDList} of labels
     * @param predictions a {@code NDList} of predictions
     */
    public abstract void update(NDList labels, NDList predictions);

    /** Resets metric values. */
    public abstract void reset();

    /**
     * Gets the name of this {@code TrainingMetric}.
     *
     * @return the name of this {@code TrainingMetric}
     */
    public String getName() {
        return name;
    }

    /**
     * Calculates metric values.
     *
     * @return a {@link Pair} of metric name and value
     */
    public abstract float getValue();

    /**
     * Checks if the two input {@code NDArray} have the same length or shape.
     *
     * @param labels a {@code NDArray} of labels
     * @param predictions a {@code NDArray} of predictions
     * @param checkDimOnly whether to check for first dimension only
     */
    protected void checkLabelShapes(NDArray labels, NDArray predictions, boolean checkDimOnly) {
        if (labels.getShape().get(0) != predictions.getShape().get(0)) {
            throw new IllegalArgumentException(
                    "The size of labels("
                            + labels.size()
                            + ") does not match that of predictions("
                            + predictions.size()
                            + ")");
        }
        if (!checkDimOnly) {
            if (labels.getShape() != predictions.getShape()) {
                throw new IllegalArgumentException(
                        "The shape of labels("
                                + labels.getShape()
                                + ") does not match that of predictions("
                                + predictions.getShape()
                                + ")");
            }
        }
    }

    /**
     * Checks the length of NDArrays.
     *
     * @param labels a {@code NDArray} of labels
     * @param predictions a {@code NDArray} of predictions
     */
    protected void checkLabelShapes(NDArray labels, NDArray predictions) {
        checkLabelShapes(labels, predictions, true);
    }
}
