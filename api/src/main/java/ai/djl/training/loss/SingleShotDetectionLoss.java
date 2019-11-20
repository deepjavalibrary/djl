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

/**
 * {@code SingleShotDetectionLoss} is an implementation of {@link Loss}. It is used to compute the
 * loss while training a Single Shot Detection (SSD) model for object detection. It involves
 * computing the targets given the generated anchors, labels and predictions, and then computing the
 * sum of class predictions and bounding box predictions.
 */
public class SingleShotDetectionLoss extends Loss {
    private Loss softmaxLoss = Loss.softmaxCrossEntropyLoss();
    private Loss l1Loss = Loss.l1Loss();
    private MultiBoxTarget multiBoxTarget = new MultiBoxTarget.Builder().build();

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name The display name of the Loss
     */
    public SingleShotDetectionLoss(String name) {
        super(name);
    }

    /**
     * Calculate loss between label and prediction.
     *
     * <p>the default implementation is simply adding all losses together
     *
     * @param labels target labels. Must contain (offsetLabels, masks, classlabels). This is
     *     returned by MultiBoxTarget function
     * @param predictions predicted labels (class prediction, offset prediction)
     * @return loss value
     */
    @Override
    public NDArray getLoss(NDList labels, NDList predictions) {
        NDArray anchors = predictions.get(0);
        NDArray classPredictions = predictions.get(1);
        NDArray boundingBoxPredictions = predictions.get(2);
        NDList targets =
                multiBoxTarget.target(
                        new NDList(anchors, labels.head(), classPredictions.transpose(0, 2, 1)));
        NDArray boundingBoxLabels = targets.get(0);
        NDArray boundingBoxMasks = targets.get(1);
        NDArray classLabels = targets.get(2);

        NDArray classLoss =
                softmaxLoss.getLoss(new NDList(classLabels), new NDList(classPredictions));
        NDArray boundingBoxLoss =
                l1Loss.getLoss(
                        new NDList(boundingBoxLabels.mul(boundingBoxMasks)),
                        new NDList(boundingBoxPredictions.mul(boundingBoxMasks)));
        return classLoss.add(boundingBoxLoss);
    }
}
