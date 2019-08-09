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
import software.amazon.ai.util.Pair;

/** Base class for all training metrics. */
abstract class TrainingMetrics {

    private String name;

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name String, name of the metric
     */
    public TrainingMetrics(String name) {
        this.name = name;
    }

    /**
     * Update training metrics based on {@link NDList} of labels and predictions.
     *
     * @param labels {@code NDList} of labels
     * @param predictions {@code NDList} of predictions
     */
    protected abstract void update(NDList labels, NDList predictions);

    /**
     * Update training metrics based on {@link NDArray} of labels and predictions.
     *
     * @param labels {@code NDArray} of labels
     * @param predictions {@code NDArray} of predictions
     */
    protected abstract void update(NDArray labels, NDArray predictions);

    /**
     * Update training metric based on {@code NDArray} of loss.
     *
     * @param loss {@code NDArray} of loss
     */
    protected abstract void update(NDArray loss);

    /** reset metric values. */
    public abstract void reset();

    /**
     * calculate metric values.
     *
     * @return {@link Pair} of metric name and value
     */
    public abstract Pair<String, Float> getMetric();

    public String getName() {
        return name;
    }

    /**
     * Check if the two input {@code NDArray} have the same length or shape.
     *
     * @param labels {@code NDArray} of labels
     * @param predictions {@code NDArray} of predictions
     * @param checkDimOnly whether to check for first dimension only
     */
    protected void checkLabelShapes(NDArray labels, NDArray predictions, boolean checkDimOnly) {
        if (labels.getShape().get(0) != predictions.getShape().get(0)) {
            throw new IllegalArgumentException(
                    String.format(
                            "The size of labels(%d) does not match that of predictions(%d)",
                            labels.size(), predictions.size()));
        }
        if (!checkDimOnly) {
            if (labels.getShape() != predictions.getShape()) {
                throw new IllegalArgumentException(
                        String.format(
                                "The shape of labels(%d) does not match that of predictions(%d)",
                                labels.getShape(), predictions.getShape()));
            }
        }
    }

    /**
     * Convenient method for checking length of NDArrays.
     *
     * @param labels {@code NDArray} of labels
     * @param predictions {@code NDArray} of predictions
     */
    protected void checkLabelShapes(NDArray labels, NDArray predictions) {
        checkLabelShapes(labels, predictions, true);
    }
}
