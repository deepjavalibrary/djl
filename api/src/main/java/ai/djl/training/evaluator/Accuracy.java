/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Pair;

/** {@link Accuracy} is the {@link AbstractAccuracy} with multiple classes. */
public class Accuracy extends AbstractAccuracy {

    /**
     * Creates a multiclass accuracy evaluator that computes accuracy across axis 1 along the 0th
     * index.
     */
    public Accuracy() {
        this("Accuracy", 1);
    }

    /**
     * Creates a multiclass accuracy evaluator that computes accuracy across axis 1 along the 0th
     * index.
     *
     * @param name the name of the evaluator, default is "Accuracy"
     */
    public Accuracy(String name) {
        this(name, 1);
    }

    /**
     * Creates a multiclass accuracy evaluator.
     *
     * @param name the name of the evaluator, default is "Accuracy"
     * @param axis the axis that represent classes in prediction, default 1
     */
    public Accuracy(String name, int axis) {
        super(name, axis);
    }

    /** {@inheritDoc} */
    @Override
    protected Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions) {
        NDArray label = labels.head();
        NDArray prediction = predictions.head();
        checkLabelShapes(label, prediction);
        NDArray predictionReduced;
        if (!label.getShape().equals(prediction.getShape())) {
            // Multi-class, sparse label
            predictionReduced = prediction.argMax(axis);
            predictionReduced = predictionReduced.reshape(label.getShape());
        } else {
            // Multi-class, one-hot label
            predictionReduced = prediction.argMax(axis);
            label = label.argMax(axis);
        }
        // result of sum is int64 now
        long total = label.size();
        try (NDArray nd = label.toType(DataType.INT64, true)) {
            NDArray correct = predictionReduced.toType(DataType.INT64, false).eq(nd).countNonzero();
            return new Pair<>(total, correct);
        }
    }
}
