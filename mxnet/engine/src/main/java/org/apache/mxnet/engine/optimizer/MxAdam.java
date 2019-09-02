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

import org.apache.mxnet.engine.MxOpParams;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.optimizer.Adam;
import software.amazon.ai.util.PairList;

public class MxAdam extends MxOptimizer implements Adam {

    private float learningRate;
    private float beta1;
    private float beta2;
    private float epsilon;
    private boolean lazyUpdate;

    private NDList means;
    private NDList variances;

    public MxAdam(Adam.Builder builder) {
        super(builder);
        learningRate = builder.getLearningRate();
        beta1 = builder.getBeta1();
        beta2 = builder.getBeta2();
        epsilon = builder.getEpsilon();
        lazyUpdate = builder.isLazyUpdate();
    }

    @Override
    public boolean initializeStates(PairList<String, Parameter> parameters) {
        if (means == null) {
            means = new NDList(parameters.size());
            variances = new NDList(parameters.size());
            for (Parameter param : parameters.values()) {
                means.add(param.getArray().zerosLike());
                variances.add(param.getArray().zerosLike());
            }
        }
        return true;
    }

    @Override
    void update(int index, NDArray weight, NDArray grad) {
        double t = updateCount(index);
        double coef1 = 1.0 - Math.pow(beta1, t);
        double coef2 = 1.0 - Math.pow(beta2, t);
        double lr = learningRate * Math.sqrt(coef2) / coef1;

        MxOpParams params = new MxOpParams();
        params.addParam("wd", getWeightDecay(index));
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("lr", lr);
        params.addParam("beta1", beta1);
        params.addParam("beta2", beta2);
        params.addParam("epsilon", epsilon);
        params.addParam("lazy_update", lazyUpdate);

        NDList inputs = new NDList(weight, grad, means.get(index), variances.get(index));
        weight.getManager().invoke("adam_update", inputs, new NDList(weight), params);
    }
}
