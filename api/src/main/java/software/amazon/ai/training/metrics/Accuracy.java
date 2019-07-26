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

/**
 * Computes accuracy classification score.
 *
 * <p>The accuracy score is defined as .. math:: \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n}
 * \\sum_{i=0}^{n-1} \\text{1}(\\hat{y_i} == y_i)
 */
public class Accuracy extends TrainingMetrics {

    private int axis;

    /**
     * Creates Accuracy metric.
     *
     * @param name name of the metric, default is "Accuracy"
     * @param axis axis used for performing argmax on prediction result
     */
    public Accuracy(String name, int axis) {
        super(name);
        this.axis = axis;
    }

    public Accuracy() {
        this("Accuracy", 1);
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDList labels, NDList predictions) {
        // number of labels and predictions should be the same
        checkLabelShapes(labels, predictions);

        for (int i = 0; i < labels.size(); i++) {
            NDArray label = labels.get(i);
            NDArray prediction = predictions.get(i);
            NDArray predictionReduced;
            if (prediction.getShape() != label.getShape()) {
                // axis always 0 for calculating predictions in NDList
                predictionReduced = prediction.argmax(0, false);
            } else {
                predictionReduced = prediction;
            }
            float[] labelArray = label.toFloatArray();
            float[] predictionArray = predictionReduced.toFloatArray();
            assert labelArray.length == predictionArray.length;
            for (int j = 0; j < labelArray.length; j++) {
                if (labelArray[j] == predictionArray[j]) {
                    addNumInstances(1);
                }
            }
        }
        addTotalInstances(labels.size());
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
        int numCorrect = (int) labels.eq(predictionReduced).sum().getFloat();
        addNumInstances(numCorrect);
        addTotalInstances(labels.size());
    }
}
