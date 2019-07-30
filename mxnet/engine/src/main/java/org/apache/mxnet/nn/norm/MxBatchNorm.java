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
package org.apache.mxnet.nn.norm;

import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.Parameter;
import software.amazon.ai.ParameterType;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.norm.BatchNorm;

public class MxBatchNorm extends MxNNBlock implements BatchNorm {

    private int axis;
    private float epsilon;
    private float momentum;
    private long inChannels;

    private Parameter runningMean;
    private Parameter runningVar;

    public MxBatchNorm(int axis, float epsilon, float momentum) {
        this.opName = "BatchNorm";
        this.axis = axis;
        this.epsilon = epsilon;
        this.momentum = momentum;
        runningMean = new Parameter("runningMean", this, ParameterType.OTHER);
        runningVar = new Parameter("runningVar", this, ParameterType.OTHER);
    }

    @Override
    public NDArray forward(NDArray data) {
        ensureInitialized(new NDList(data));
        NDArray gamma = data.getManager().ones(new Shape(inChannels));
        NDArray beta = data.getManager().zeros(new Shape(inChannels));
        NDList inputs =
                new NDList(data, gamma, beta, runningMean.getArray(), runningVar.getArray());
        MxOpParams params = new MxOpParams();
        params.addParam("eps", epsilon);
        params.addParam("momentum", momentum);
        params.addParam("axis", axis);
        return forward(inputs, params).get(0);
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return inputs[0];
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Arrays.asList(runningMean, runningVar);
    }

    @Override
    public void beforeInitialize(NDList inputs) {
        inChannels = inputs.get(0).size(axis);
    }

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        switch (name) {
            case "runningMean":
            case "runningVar":
                return new Shape(inChannels);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }
}
