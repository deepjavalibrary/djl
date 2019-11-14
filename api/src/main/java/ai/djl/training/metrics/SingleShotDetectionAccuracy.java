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
import ai.djl.ndarray.types.DataType;

public class SingleShotDetectionAccuracy extends Accuracy {
    private MultiBoxTarget multiBoxTarget = new MultiBoxTarget.Builder().build();

    public SingleShotDetectionAccuracy(String name) {
        super(name, 0);
    }

    @Override
    public void update(NDList labels, NDList predictions) {
        NDArray anchors = predictions.get(0);
        NDArray classPredictions = predictions.get(1);
        NDList targets =
                multiBoxTarget.target(
                        new NDList(anchors, labels.head(), classPredictions.transpose(0, 2, 1)));
        NDArray classLabels = targets.get(2);
        checkLabelShapes(classLabels, classPredictions);
        NDArray predictionReduced = classPredictions.argMax(-1);
        long numCorrect =
                classLabels
                        .asType(DataType.INT64, false)
                        .eq(predictionReduced.asType(DataType.INT64, false))
                        .countNonzero();
        addCorrectInstances(numCorrect);
        addTotalInstances(classLabels.size());
    }
}
