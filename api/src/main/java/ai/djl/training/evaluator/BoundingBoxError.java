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

import ai.djl.modality.cv.MultiBoxTarget;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code BoundingBoxError} is an {@link Evaluator} that computes the error in the prediction of
 * bounding boxes in SingleShotDetection model.
 */
public class BoundingBoxError extends Evaluator {

    private Map<String, Float> ssdBoxPredictionError;
    private MultiBoxTarget multiBoxTarget = MultiBoxTarget.builder().build();

    /**
     * Creates an BoundingBoxError evaluator.
     *
     * @param name the name of the evaluator
     */
    public BoundingBoxError(String name) {
        super(name);
        ssdBoxPredictionError = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray anchors = predictions.get(0);
        NDArray classPredictions = predictions.get(1);
        NDArray boundingBoxPredictions = predictions.get(2);
        NDList targets =
                multiBoxTarget.target(
                        new NDList(anchors, labels.head(), classPredictions.transpose(0, 2, 1)));
        NDArray boundingBoxLabels = targets.get(0);
        NDArray boundingBoxMasks = targets.get(1);
        return boundingBoxLabels.sub(boundingBoxPredictions).mul(boundingBoxMasks).abs();
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        totalInstances.put(key, 0L);
        ssdBoxPredictionError.put(key, 0f);
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        NDArray boundingBoxError = evaluate(labels, predictions);
        float update = boundingBoxError.sum().getFloat();
        totalInstances.compute(key, (k, v) -> v + boundingBoxError.size());
        ssdBoxPredictionError.compute(key, (k, v) -> v + update);
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        totalInstances.compute(key, (k, v) -> 0L);
        ssdBoxPredictionError.compute(key, (k, v) -> 0f);
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        Long total = totalInstances.get(key);
        Objects.requireNonNull(total, "No evaluator found at that path");

        if (total == 0) {
            return Float.NaN;
        }

        return ssdBoxPredictionError.get(key) / totalInstances.get(key);
    }
}
