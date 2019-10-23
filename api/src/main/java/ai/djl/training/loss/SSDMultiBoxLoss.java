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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

public class SSDMultiBoxLoss extends Loss {
    Loss softmaxLoss = Loss.softmaxCrossEntropyLoss();
    Loss l1Loss = Loss.l1Loss();

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name The display name of the Loss
     */
    public SSDMultiBoxLoss(String name) {
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
        NDArray offsetLabels = labels.head();
        NDArray masks = labels.get(1);
        NDArray classLabels = labels.get(2);

        NDArray classPredictions = predictions.get(0);
        NDArray offsetPredictions = predictions.get(1);

        NDArray classLoss =
                softmaxLoss.getLoss(new NDList(classLabels), new NDList(classPredictions));
        NDArray bBoxLoss =
                l1Loss.getLoss(
                        new NDList(offsetLabels.mul(masks)),
                        new NDList(offsetPredictions.mul(masks)));
        return classLoss.add(bBoxLoss);
    }
}
