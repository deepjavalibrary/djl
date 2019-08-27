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
package org.apache.mxnet.nn.convolutional;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.convolutional.Conv1D;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.PairList;

public class MxConv1D extends MxNNBlock implements Conv1D {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.WIDTH
    };

    private static final String LAYOUT_STRING = "NCW";
    private static final byte VERSION = 1;

    private Shape kernel;
    private Shape stride;
    private Shape pad;
    private Shape dilate;
    private int numFilters;
    private int numGroups;
    private boolean includeBias;

    private Parameter weight;
    private Parameter bias;

    public MxConv1D(Conv1D.Builder builder) {
        opName = "Convolution";
        kernel = builder.getKernel();
        stride = builder.getStride() == null ? new Shape(1) : builder.getStride();
        pad = builder.getPad() == null ? new Shape(0) : builder.getPad();
        dilate = builder.getDilate() == null ? new Shape(1) : builder.getDilate();
        numFilters = builder.getNumFilters();
        numGroups = builder.getNumGroups();
        includeBias = builder.isIncludeBias();
        weight = new Parameter("weight", this, ParameterType.WEIGHT);
        if (includeBias) {
            bias = new Parameter("bias", this, ParameterType.BIAS, Initializer.ZEROS);
        }
    }

    @Override
    public Shape getOutputShape(final Shape... inputs) {
        long batchSize = inputs[0].get(0);
        long outWidth =
                (inputs[0].get(2) + 2 * pad.get(0) - dilate.get(0) * (kernel.get(0) - 1) - 1)
                                / stride.get(0)
                        + 1;
        return new Shape(batchSize, numFilters, outWidth);
    }

    @Override
    public List<Parameter> getDirectParameters() {
        List<Parameter> parameters = new ArrayList<>();
        parameters.add(weight);
        if (includeBias) {
            parameters.add(bias);
        }
        return parameters;
    }

    @Override
    public void beforeInitialize(final NDList inputs) {
        NDArray input = inputs.head();
        Shape inputShape = input.getShape();
        if (!isLayoutSupported(EXPECTED_LAYOUT, inputShape.getLayout())) {
            throw new UnsupportedOperationException("Conv1D requires NCW layout");
        }
    }

    @Override
    public Shape getParameterShape(final String name, final NDList inputs) {
        NDArray input = inputs.head();
        Shape inputShape = input.getShape();
        switch (name) {
            case "weight":
                return new Shape(numFilters, inputShape.get(1), kernel.get(0));
            case "bias":
                return new Shape(numFilters);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    @Override
    public NDArray forward(final NDArray data) {
        return forward(new NDList(data)).get(0);
    }

    @Override
    protected NDList opInputs(NDList inputs) {
        NDArray data = inputs.get(0);
        return bias != null
                ? new NDList(data, weight.getArray(), bias.getArray())
                : new NDList(data, weight.getArray());
    }

    @Override
    protected PairList<String, Object> opParams(PairList<String, Object> params) {
        MxOpParams result = new MxOpParams();
        result.addParam("kernel", kernel);
        result.addParam("stride", stride);
        result.addParam("pad", pad);
        result.addParam("num_filter", numFilters);
        result.addParam("num_group", numGroups);
        result.add("layout", LAYOUT_STRING);
        result.add("no_bias", !includeBias);
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
