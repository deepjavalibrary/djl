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
import software.amazon.ai.util.PairList;

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

    /** {@inheritDoc} */
    @Override
    public Shape getOutputShape(Shape... inputs) {
        return inputs[0];
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Arrays.asList(runningMean, runningVar);
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(NDList inputs) {
        inChannels = inputs.get(0).size(axis);
    }

    /** {@inheritDoc} */
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

    /** {@inheritDoc} */
    @Override
    public NDArray forward(NDArray data) {
        return forward(new NDList(data)).get(0);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList opInputs(NDList inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Linear requires exactly 1 NDArray");
        }
        NDArray data = inputs.get(0);
        NDArray gamma = data.getManager().ones(new Shape(inChannels));
        NDArray beta = data.getManager().zeros(new Shape(inChannels));
        return new NDList(data, gamma, beta, runningMean.getArray(), runningVar.getArray());
    }

    /** {@inheritDoc} */
    @Override
    protected PairList<String, Object> opParams(PairList<String, Object> params) {
        MxOpParams result = new MxOpParams();
        result.addParam("eps", epsilon);
        result.addParam("momentum", momentum);
        result.addParam("axis", axis);
        result.addAll(params);
        return result;
    }
}
