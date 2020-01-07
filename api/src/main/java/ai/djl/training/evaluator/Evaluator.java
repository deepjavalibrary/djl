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
import ai.djl.util.Pair;

/**
 * Base class for all {@code Evaluator}s that can be used to evaluate the performance of a model.
 *
 * <p>The {@code Evaluator}s can all be monitored to make an assessment about the performance of the
 * model. However, only ones that further extend {@link ai.djl.training.loss.Loss} are suited to
 * being used to directly optimize a model.
 */
public abstract class Evaluator implements Cloneable {

    private String name;

    /**
     * Creates an evaluator with abstract update methods.
     *
     * @param name the name of the evaluator
     */
    public Evaluator(String name) {
        this.name = name;
    }

    /**
     * Creates and returns a copy of this object.
     *
     * @return a copy of this {@code Evaluator}
     */
    public Evaluator duplicate() {
        try {
            return (Evaluator) clone();
        } catch (CloneNotSupportedException e) {
            // ignore
            throw new AssertionError("Clone is not supported", e);
        }
    }

    /**
     * Computes and updates the evaluators based on the labels and predictions.
     *
     * @param labels a {@code NDList} of labels
     * @param predictions a {@code NDList} of predictions
     */
    public abstract void update(NDList labels, NDList predictions);

    /** Resets evaluator values. */
    public abstract void reset();

    /**
     * Gets the name of this {@code Evaluator}.
     *
     * @return the name of this {@code Evaluator}
     */
    public String getName() {
        return name;
    }

    /**
     * Calculates evaluator values.
     *
     * @return a {@link Pair} of evaluator name and value
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
