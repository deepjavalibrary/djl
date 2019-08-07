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

import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.training.Gradient.OptimizerGrad;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.lrscheduler.LrScheduler;
import software.amazon.ai.util.PairList;

/** MXNet helper containing base implementations for optimizers. */
public abstract class MxOptimizer implements Optimizer {

    PairList<String, Parameter> parameters;
    float rescaleGrad;
    float clipGrad;
    float weightDecays;
    private LrScheduler lrscheduler;
    private int numUpdate;

    public MxOptimizer(
            PairList<String, Parameter> parameters,
            float rescaleGrad,
            float weightDecays,
            float clipGrad,
            LrScheduler lrScheduler,
            int beginNumUpdate) {
        this.parameters = parameters;
        this.rescaleGrad = rescaleGrad;
        this.lrscheduler = lrScheduler;
        this.weightDecays = weightDecays;
        this.clipGrad = clipGrad;
        this.numUpdate = beginNumUpdate;
    }

    @Override
    public void step(OptimizerGrad grads) {
        PairList<String, NDArray> paramGrads = grads.get();
        for (int i = 0; i < parameters.size(); i++) {
            NDArray paramArray = parameters.get(i).getValue().getArray();
            NDArray grad = paramGrads.get(i).getValue();
            update(i, paramArray, grad);
        }
    }

    public float getLearningRate() {
        return lrscheduler.getNewLearningRate(numUpdate);
    }

    @Override
    public PairList<String, Parameter> getParameters() {
        return parameters;
    }

    public abstract void update(int index, NDArray weight, NDArray grad);
}
