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
package org.apache.mxnet.nn.core;

import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.Parameter;
import software.amazon.ai.ParameterType;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.util.Pair;

public class MxLinear extends MxNNBlock implements Linear {

    private long outChannels;

    private Parameter weight;
    private Parameter bias;

    public MxLinear(long outChannels, boolean bias) {
        this.opName = "FullyConnected";
        this.outChannels = outChannels;
        weight = new Parameter("weight", this, ParameterType.WEIGHT);
        if (bias) {
            this.bias = new Parameter("bias", this, ParameterType.BIAS);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape getOutputShape(Shape... inputs) {
        return new Shape(inputs[0].get(0), outChannels);
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        if (bias != null) {
            return Arrays.asList(weight, bias);
        }
        return Arrays.asList(weight);
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(NDList inputs) {
        Shape input = inputs.head().getShape();
        if (input.isLayoutKnown()) {
            inChannels = input.filterByLayoutType(t -> !t.equals(LayoutType.BATCH));
            inputShape =
                    input.map(
                            pair ->
                                    new Pair<>(
                                            pair.getValue().equals(LayoutType.BATCH)
                                                    ? Long.valueOf(-1)
                                                    : pair.getKey(),
                                            pair.getValue()));
        } else if (input.dimension() > 1) {
            inChannels = input.slice(1);
            inputShape =
                    new Shape(new long[] {-1}, new LayoutType[] {LayoutType.BATCH})
                            .addAll(input.slice(1));
        } else {
            inChannels = input.slice(0);
            inputShape = input;
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        switch (name) {
            case "weight":
                return new Shape(outChannels).addAll(inChannels);
            case "bias":
                return new Shape(outChannels);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray forward(NDArray data) {
        ensureInitialized(new NDList(data));
        NDList inputs =
                bias != null
                        ? new NDList(data, weight.getArray(), bias.getArray())
                        : new NDList(data, weight.getArray());
        MxOpParams params = new MxOpParams();
        params.addParam("num_hidden", outChannels);
        params.addParam("flatten", false);
        params.addParam("no_bias", bias == null);
        return forward(inputs, params).get(0);
    }
}
