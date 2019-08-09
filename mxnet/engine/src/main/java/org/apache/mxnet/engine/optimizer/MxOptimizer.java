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

import java.util.Arrays;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.training.Gradient.OptimizerGrad;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.PairList;

/** MXNet helper containing base implementations for optimizers. */
public abstract class MxOptimizer implements Optimizer {

    PairList<String, Parameter> parameters;
    float rescaleGrad;
    float clipGrad;
    private float weightDecays;
    int numUpdate;

    private int[] updateCounts;

    public MxOptimizer(BaseBuilder<?> builder) {
        this.parameters = builder.getParameters();
        this.rescaleGrad = builder.getRescaleGrad();
        this.weightDecays = builder.getWeightDecays();
        this.clipGrad = builder.getClipGrad();
        this.numUpdate = builder.getBeginNumUpdate();

        if (rescaleGrad == 0) {
            throw new IllegalArgumentException("The rescaleGrad should be set");
        }

        updateCounts = new int[parameters.size()];
        Arrays.fill(updateCounts, numUpdate);
    }

    @Override
    public void step(OptimizerGrad grads) {
        PairList<String, NDArray> paramGrads = grads.get();
        for (int i = 0; i < parameters.size(); i++) {
            NDArray paramArray = parameters.get(i).getValue().getArray();
            NDArray grad = paramGrads.get(i).getValue();
            update(i, paramArray, grad);
        }
        numUpdate++;
    }

    @Override
    public PairList<String, Parameter> getParameters() {
        return parameters;
    }

    float getWeightDecay(int index) {
        return weightDecays;
    }

    int updateCount(int index) {
        int count = ++updateCounts[index];
        numUpdate = Math.max(numUpdate, count);
        return count;
    }

    public abstract void update(int index, NDArray weight, NDArray grad);
}
