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
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code Accuracy} is an {@link Evaluator} that computes the accuracy score.
 *
 * <p>The accuracy score is defined as \(accuracy(y, \hat{y}) =
 * \frac{1}{n}\sum_{i=0}^{n-1}1(\hat{y_i} == y_i)\)
 */
public abstract class AbstractAccuracy extends Evaluator {

    protected Map<String, Long> correctInstances;
    protected int axis;
    protected int index;

    /**
     * Creates an accuracy evaluator that computes accuracy across axis 1 along given index.
     *
     * @param name the name of the evaluator, default is "Accuracy"
     * @param index the index of the NDArray in labels to compute accuracy for
     */
    public AbstractAccuracy(String name, int index) {
        this(name, index, 1);
    }

    /**
     * Creates an accuracy evaluator.
     *
     * @param name the name of the evaluator, default is "Accuracy"
     * @param index the index of the NDArray in labels to compute accuracy for
     * @param axis the axis that represent classes in prediction, default 1
     */
    public AbstractAccuracy(String name, int index, int axis) {
        super(name);
        correctInstances = new ConcurrentHashMap<>();
        this.axis = axis;
        this.index = index;
    }

    /**
     * A helper for classes extending {@link AbstractAccuracy}.
     *
     * @param labels the labels to get accuracy for
     * @param predictions the predictions to get accuracy for
     * @return a pair(number of total values, ndarray int of correct values)
     */
    protected abstract Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions);

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        return accuracyHelper(labels, predictions).getValue();
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        totalInstances.put(key, 0L);
        correctInstances.put(key, 0L);
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        Pair<Long, NDArray> update = accuracyHelper(labels, predictions);
        totalInstances.compute(key, (k, v) -> v + update.getKey());
        correctInstances.compute(key, (k, v) -> v + update.getValue().sum().getLong());
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        totalInstances.compute(key, (k, v) -> 0L);
        correctInstances.compute(key, (k, v) -> 0L);
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        Long total = totalInstances.get(key);
        if (total == null || total == 0) {
            return Float.NaN;
        }

        return (float) correctInstances.get(key) / totalInstances.get(key);
    }
}
