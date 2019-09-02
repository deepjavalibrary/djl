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
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.PairList;

/** MXNet helper containing base implementations for optimizers. */
public abstract class MxOptimizer implements Optimizer {

    float rescaleGrad;
    float clipGrad;
    private float weightDecays;
    int numUpdate;
    private boolean statesInitialized;
    private int[] updateCounts;

    public MxOptimizer(BaseBuilder<?> builder) {
        this.rescaleGrad = builder.getRescaleGrad();
        this.weightDecays = builder.getWeightDecays();
        this.clipGrad = builder.getClipGrad();
        this.numUpdate = builder.getBeginNumUpdate();

        if (rescaleGrad == 0) {
            throw new IllegalArgumentException("The rescaleGrad should be set");
        }
    }

    @Override
    public void updateAllParameters(PairList<String, Parameter> parameters) {
        if (!statesInitialized) {
            // ensure when create state is over ridden, statesCreated is updated
            statesInitialized = initializeStates(parameters);
        }
        if (updateCounts == null) {
            updateCounts = new int[parameters.size()];
            Arrays.fill(updateCounts, numUpdate);
        }
        for (int i = 0; i < parameters.size(); i++) {
            NDArray paramArray = parameters.get(i).getValue().getArray();
            NDArray grad = paramArray.getGradient();
            update(i, paramArray, grad);
        }
    }

    float getWeightDecay(int index) {
        return weightDecays;
    }

    int updateCount(int index) {
        int count = ++updateCounts[index];
        numUpdate = Math.max(numUpdate, count);
        return count;
    }

    abstract void update(int index, NDArray weight, NDArray grad);

    abstract boolean initializeStates(PairList<String, Parameter> parameters);
}
