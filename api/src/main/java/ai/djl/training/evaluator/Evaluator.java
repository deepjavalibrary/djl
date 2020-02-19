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
package ai.djl.training.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base class for all {@code Evaluator}s that can be used to evaluate the performance of a model.
 *
 * <p>The {@code Evaluator}s can all be monitored to make an assessment about the performance of the
 * model. However, only ones that further extend {@link ai.djl.training.loss.Loss} are suited to
 * being used to directly optimize a model.
 *
 * <p>In addition to computing the evaluation, an evaluator can accumulate values to compute a total
 * evaluation. For different purposes, it is possible to have multiple accumulators on a single
 * evaluator. Each accumulator must be added with a String key to identify the accumulator. Before
 * using an accumulator, you must {@link Evaluator#addAccumulator(String)}. Then, call {@link
 * Evaluator#updateAccumulator(String, NDList, NDList)} to add more data to the accumulator. You can
 * use {@link Evaluator#getAccumulator(String)} to retrieve the accumulated value and {@link
 * Evaluator#resetAccumulator(String)} to reset the accumulator to the same value as when just
 * added.
 */
public abstract class Evaluator {

    private String name;
    protected Map<String, Long> totalInstances;

    /**
     * Creates an evaluator with abstract update methods.
     *
     * @param name the name of the evaluator
     */
    public Evaluator(String name) {
        this.name = name;
        totalInstances = new ConcurrentHashMap<>();
    }

    /**
     * Returns the name of this {@code Evaluator}.
     *
     * @return the name of this {@code Evaluator}
     */
    public String getName() {
        return name;
    }

    /**
     * Calculates the evaluation between the labels and the predictions.
     *
     * @param labels the correct values
     * @param predictions the predicted values
     * @return the evaluation result
     */
    public abstract NDArray evaluate(NDList labels, NDList predictions);

    /**
     * Adds an accumulator for the results of the evaluation with the given key.
     *
     * @param key the key for the new accumulator
     */
    public abstract void addAccumulator(String key);

    /**
     * Updates the evaluator with the given key based on a {@link NDList} of labels and predictions.
     *
     * <p>This is a synchronized operation. You should only call it at the end of a batch or epoch.
     *
     * @param key the key of the accumulator to update
     * @param labels a {@code NDList} of labels
     * @param predictions a {@code NDList} of predictions
     */
    public abstract void updateAccumulator(String key, NDList labels, NDList predictions);

    /**
     * Resets the evaluator value with the given key.
     *
     * @param key the key of the accumulator to reset
     */
    public abstract void resetAccumulator(String key);

    /**
     * Returns the accumulated evaluator value.
     *
     * @param key the key of the accumulator to get
     * @return the accumulated value
     * @throws IllegalArgumentException if no accumulator was added with the given key
     */
    public abstract float getAccumulator(String key);

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
            if (!labels.getShape().equals(predictions.getShape())) {
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
