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

import ai.djl.modality.cv.MultiBoxTarget;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/**
 * {@code BoundingBoxError} is a {@link TrainingMetric} that computes the error in the prediction of
 * bounding boxes in SingleShotDetection model.
 */
public class BoundingBoxError extends TrainingMetric {

    private float ssdBoxPredictionError;
    private float numInstances;
    private MultiBoxTarget multiBoxTarget = new MultiBoxTarget.Builder().build();

    /**
     * Creates an BoundingBoxError metric.
     *
     * @param name the name of the metric.
     */
    public BoundingBoxError(String name) {
        super(name);
    }

    /**
     * Computes and updates the detection bounding box prediction error based on {@link NDList} of
     * labels and predictions.
     *
     * <p>First compute bounding box labels and bounding box masks using the MultiBoxTarget
     * operator. Then compute bounding box error based on bounding box labels and predictions.
     *
     * @param labels the {@code NDList} of labels
     * @param predictions the {@code NDList} of predictions
     */
    @Override
    public void update(NDList labels, NDList predictions) {
        NDArray anchors = predictions.get(0);
        NDArray classPredictions = predictions.get(1);
        NDArray boundingBoxPredictions = predictions.get(2);
        NDList targets =
                multiBoxTarget.target(
                        new NDList(anchors, labels.head(), classPredictions.transpose(0, 2, 1)));
        NDArray boundingBoxLabels = targets.get(0);
        NDArray boundingBoxMasks = targets.get(1);
        NDArray boundingBoxError =
                boundingBoxLabels.sub(boundingBoxPredictions).mul(boundingBoxMasks).abs().sum();
        ssdBoxPredictionError += boundingBoxError.getFloat();
        numInstances += boundingBoxLabels.size();
    }

    @Override
    public TrainingMetric duplicate() {
        return new BoundingBoxError(getName());
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        this.ssdBoxPredictionError = 0;
        this.numInstances = 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getValue() {
        return ssdBoxPredictionError / numInstances;
    }
}
