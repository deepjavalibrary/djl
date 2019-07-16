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

    private int units;
    private Shape inUnits;

    private Parameter weight;
    private Parameter bias;

    public MxLinear(int units) {
        this.opName = "FullyConnected";
        this.units = units;
        weight = new Parameter("weight", this, ParameterType.WEIGHT);
        bias = new Parameter("bias", this, ParameterType.BIAS);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getOutputShape(Shape... inputs) {
        return new Shape(inputs[0].get(0), units);
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Arrays.asList(weight, bias);
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(NDList inputs) {
        Shape input = inputs.head().getShape();
        inUnits = input.filterByLayoutType(t -> !t.equals(LayoutType.BATCH));
        inputShape =
                input.map(
                        pair ->
                                new Pair<>(
                                        pair.getValue().equals(LayoutType.BATCH)
                                                ? Long.valueOf(-1)
                                                : pair.getKey(),
                                        pair.getValue()));
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        switch (name) {
            case "weight":
                return new Shape(units).addAll(inUnits);
            case "bias":
                return new Shape(units);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray forward(NDArray data) {
        ensureInitialized(new NDList(data));
        NDList inputs = new NDList(data, weight.getArray(), bias.getArray());
        MxOpParams params = new MxOpParams();
        params.add("num_hidden", "1");
        return forward(inputs, params).get(0);
    }
}
