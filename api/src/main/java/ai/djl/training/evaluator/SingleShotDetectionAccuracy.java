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
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Pair;

/**
 * {@code SingleShotDetectionAccuracy} is an implementation of {@link AbstractAccuracy}. It is used
 * while training a Single Shot Detection (SSD) model for object detection. It uses the targets
 * computed by {@link MultiBoxTarget}, and computes the class prediction accuracy against the
 * computed targets.
 */
public class SingleShotDetectionAccuracy extends AbstractAccuracy {

    private MultiBoxTarget multiBoxTarget = MultiBoxTarget.builder().build();

    /**
     * Creates a new instance of {@link SingleShotDetectionAccuracy} with the given name.
     *
     * @param name the name given to the accuracy
     */
    public SingleShotDetectionAccuracy(String name) {
        super(name, 0);
    }

    /** {@inheritDoc} */
    @Override
    protected Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions) {
        NDArray anchors = predictions.get(0);
        NDArray classPredictions = predictions.get(1);
        NDList targets =
                multiBoxTarget.target(
                        new NDList(anchors, labels.head(), classPredictions.transpose(0, 2, 1)));
        NDArray classLabels = targets.get(2);
        checkLabelShapes(classLabels, classPredictions);
        NDArray predictionReduced = classPredictions.argMax(-1);
        long total = classLabels.size();
        NDArray numCorrect =
                classLabels.toType(DataType.INT64, false).eq(predictionReduced).countNonzero();
        return new Pair<>(total, numCorrect);
    }
}
