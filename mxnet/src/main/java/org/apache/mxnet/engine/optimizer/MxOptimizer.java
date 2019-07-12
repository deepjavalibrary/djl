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
package org.apache.mxnet.engine.optimizer;

import org.apache.mxnet.engine.lrscheduler.MxLearningRateTracker;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;

public abstract class MxOptimizer {

    float rescaleGrad;
    float clipGrad;
    float weightDecays;
    private MxLearningRateTracker lrscheduler;
    private int numUpdate;

    public MxOptimizer(
            float rescaleGrad,
            float weightDecays,
            float clipGrad,
            MxLearningRateTracker lrScheduler,
            int beginNumUpdate) {
        this.rescaleGrad = rescaleGrad;
        this.lrscheduler = lrScheduler;
        this.weightDecays = weightDecays;
        this.clipGrad = clipGrad;
        this.numUpdate = beginNumUpdate;
    }

    public float getLearningRate() {
        return lrscheduler.getNewLearningRate(numUpdate);
    }

    public abstract NDList createState(int index, NDArray weight);

    public abstract void update(int index, NDArray weight, NDArray grad, NDList state);
}
