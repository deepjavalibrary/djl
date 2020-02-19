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
package ai.djl.training.loss;

import ai.djl.modality.cv.MultiBoxTarget;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;
import java.util.Arrays;

/**
 * {@code SingleShotDetectionLoss} is an implementation of {@link Loss}. It is used to compute the
 * loss while training a Single Shot Detection (SSD) model for object detection. It involves
 * computing the targets given the generated anchors, labels and predictions, and then computing the
 * sum of class predictions and bounding box predictions.
 */
public class SingleShotDetectionLoss extends AbstractCompositeLoss {

    private MultiBoxTarget multiBoxTarget = MultiBoxTarget.builder().build();

    /** Base class for metric with abstract update methods. */
    public SingleShotDetectionLoss() {
        super("SingleShotDetectionLoss");
        components =
                Arrays.asList(
                        Loss.softmaxCrossEntropyLoss("ClassLoss"), Loss.l1Loss("BoundingBoxLoss"));
    }

    /**
     * Calculate loss between label and prediction.
     *
     * @param labels target labels. Must contain (offsetLabels, masks, classlabels). This is
     *     returned by MultiBoxTarget function
     * @param predictions predicted labels (class prediction, offset prediction)
     * @return loss value
     */
    @Override
    protected Pair<NDList, NDList> inputForComponent(
            int componentIndex, NDList labels, NDList predictions) {
        NDArray anchors = predictions.get(0);
        NDArray classPredictions = predictions.get(1);
        NDList targets =
                multiBoxTarget.target(
                        new NDList(anchors, labels.head(), classPredictions.transpose(0, 2, 1)));

        switch (componentIndex) {
            case 0: // ClassLoss
                NDArray classLabels = targets.get(2);
                return new Pair<>(new NDList(classLabels), new NDList(classPredictions));
            case 1: // BoundingBoxLoss
                NDArray boundingBoxPredictions = predictions.get(2);
                NDArray boundingBoxLabels = targets.get(0);
                NDArray boundingBoxMasks = targets.get(1);
                return new Pair<>(
                        new NDList(boundingBoxLabels.mul(boundingBoxMasks)),
                        new NDList(boundingBoxPredictions.mul(boundingBoxMasks)));
            default:
                throw new IllegalArgumentException("Invalid component index");
        }
    }
}
