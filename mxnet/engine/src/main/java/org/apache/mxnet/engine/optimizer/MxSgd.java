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

import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.lrscheduler.LrScheduler;
import software.amazon.ai.util.PairList;

public class MxSgd extends MxOptimizer implements Sgd {

    private float momentum;
    private boolean lazyUpdate;
    private List<NDArray> momentumStates;

    public MxSgd(
            PairList<String, Parameter> parameters,
            float rescaleGrad,
            float weightDecays,
            float clipGrad,
            LrScheduler lrScheduler,
            int beginNumUpdate,
            float momentum,
            boolean lazyUpdate) {
        super(parameters, rescaleGrad, weightDecays, clipGrad, lrScheduler, beginNumUpdate);
        this.momentum = momentum;
        this.lazyUpdate = lazyUpdate;
    }

    @Override
    public void update(int index, NDArray weight, NDArray grad) {
        // TODO: Support Mixed precision Sparse
        if (momentum != 0) {
            if (momentumStates == null) {
                momentumStates = new ArrayList<>(parameters.size());
                for (Parameter param : parameters.values()) {
                    momentumStates.add(param.getArray().zerosLike());
                }
            }
            weight.getNDArrayInternal()
                    .sgdMomUpdate(
                            grad,
                            momentumStates.get(index),
                            getLearningRate(),
                            weightDecays,
                            momentum,
                            rescaleGrad,
                            clipGrad,
                            lazyUpdate);
        } else {
            weight.getNDArrayInternal()
                    .sgdUpdate(
                            grad,
                            getLearningRate(),
                            weightDecays,
                            rescaleGrad,
                            clipGrad,
                            lazyUpdate);
        }
    }
}
