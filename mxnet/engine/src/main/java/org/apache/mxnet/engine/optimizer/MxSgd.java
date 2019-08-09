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
import org.apache.mxnet.engine.MxOpParams;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.learningrate.LrTracker;

public class MxSgd extends MxOptimizer implements Sgd {

    private LrTracker lrTracker;
    private float momentum;
    private boolean lazyUpdate;

    private List<NDArray> momentumStates;

    public MxSgd(Sgd.Builder builder) {
        super(builder);
        lrTracker = builder.getLrTracker();
        momentum = builder.getMomentum();
        lazyUpdate = builder.isLazyUpdate();
    }

    @Override
    public void update(int index, NDArray weight, NDArray grad) {
        // TODO: Support Mixed precision Sparse
        float learningRate = lrTracker.getNewLearningRate(updateCount(index));
        if (momentum != 0) {
            if (momentumStates == null) {
                momentumStates = new ArrayList<>(parameters.size());
                for (Parameter param : parameters.values()) {
                    momentumStates.add(param.getArray().zerosLike());
                }
            }
            MxOpParams params = new MxOpParams();
            params.addParam("wd", getWeightDecay(index));
            params.addParam("rescale_grad", rescaleGrad);
            params.addParam("clip_gradient", clipGrad);

            params.addParam("lr", learningRate);
            params.addParam("momentum", momentum);
            params.addParam("lazy_update", lazyUpdate);
            weight.getManager()
                    .invoke(
                            "sgd_mom_update",
                            new NDList(weight, grad, momentumStates.get(index)),
                            new NDList(weight),
                            params);
        } else {
            MxOpParams params = new MxOpParams();
            params.addParam("wd", getWeightDecay(index));
            params.addParam("rescale_grad", rescaleGrad);
            params.addParam("clip_gradient", clipGrad);

            params.addParam("lr", learningRate);
            params.addParam("lazy_update", lazyUpdate);
            weight.getManager()
                    .invoke("sgd_update", new NDList(weight, grad), new NDList(weight), params);
        }
    }
}
