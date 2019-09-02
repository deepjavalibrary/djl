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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

public class MxLinear extends MxNNBlock implements Linear {

    private static final byte VERSION = 1;

    private long outChannels;

    private Parameter weight;
    private Parameter bias;

    public MxLinear(long outChannels, boolean bias) {
        this.opName = "FullyConnected";
        this.outChannels = outChannels;
        weight = new Parameter("weight", this, ParameterType.WEIGHT);
        if (bias) {
            this.bias = new Parameter("bias", this, ParameterType.BIAS, Initializer.ZEROS);
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
        return Collections.singletonList(weight);
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
        return forward(new NDList(data)).get(0);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList opInputs(NDList inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Linear requires exactly 1 NDArray");
        }
        NDList additional =
                bias != null
                        ? new NDList(weight.getArray(), bias.getArray())
                        : new NDList(weight.getArray());
        NDList result = new NDList();
        result.addAll(inputs);
        result.addAll(additional);
        return result;
    }

    /** {@inheritDoc} */
    @Override
    protected PairList<String, Object> opParams(PairList<String, Object> params) {
        MxOpParams result = new MxOpParams();
        result.addParam("num_hidden", outChannels);
        result.addParam("flatten", false);
        result.addParam("no_bias", bias == null);
        result.addAll(params);
        return result;
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        weight.save(os);
        if (bias != null) {
            bias.save(os);
        }
    }

    @Override
    public void loadParameters(DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        weight.load(is);
        if (bias != null) {
            bias.load(is);
        }
    }
}
