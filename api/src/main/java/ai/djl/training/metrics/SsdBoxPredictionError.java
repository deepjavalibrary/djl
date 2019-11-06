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

/**
 * {@code SsdBoxPredictionError} is a {@link TrainingMetric} that computes the error in the
 * prediction of bounding boxes in SSD.
 */
public class SsdBoxPredictionError extends TrainingMetric {

    private float ssdBoxPredictionError;

    /**
     * Creates an SSDBoxPredictionError metric.
     *
     * @param name the name of the metric, default is "Accuracy"
     */
    public SsdBoxPredictionError(String name) {
        super(name);
    }

    /**
     * Computes and updates the SSD box prediction error based on {@link NDList} of labels and
     * predictions.
     *
     * <p>For SSD Box offset accuracy, targets must contain (bounding box labels, bounding box
     * masks) that are returned by the MultiBoxTarget operator.
     *
     * @param targets the {@code NDList} of targets
     * @param predictions the {@code NDList} of predictions
     */
    @Override
    public void update(NDList targets, NDList predictions) {
        NDArray boundingBoxLabels = targets.get(0);
        NDArray boundingBoxMasks = targets.get(1);
        NDArray boundingBoxPredictions = predictions.head();
        NDArray boundingBoxError =
                boundingBoxLabels.sub(boundingBoxPredictions).mul(boundingBoxMasks).abs().sum();
        ssdBoxPredictionError += boundingBoxError.getFloat();
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        this.ssdBoxPredictionError = 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getValue() {
        return ssdBoxPredictionError;
    }
}
