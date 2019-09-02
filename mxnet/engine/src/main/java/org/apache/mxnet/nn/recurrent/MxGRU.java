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
package org.apache.mxnet.nn.recurrent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.recurrent.GRU;
import software.amazon.ai.util.PairList;

public class MxGRU extends MxNNBlock implements GRU {
    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.TIME, LayoutType.BATCH, LayoutType.CHANNEL
    };
    private long stateSize;
    private float dropRate;
    private int numStackedLayers;
    private String mode;
    private boolean useSequenceLength;
    private boolean useBidirectional;
    private boolean stateOutputs;

    private List<Parameter> parameters =
            Arrays.asList(
                    new Parameter("i2rWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2rBias", this, ParameterType.BIAS),
                    new Parameter("h2rWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2rBias", this, ParameterType.BIAS),
                    new Parameter("i2zWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2zBias", this, ParameterType.BIAS),
                    new Parameter("h2zWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2zBias", this, ParameterType.BIAS),
                    new Parameter("i2nWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2nBias", this, ParameterType.BIAS),
                    new Parameter("h2nWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2nBias", this, ParameterType.BIAS));

    private Parameter state = new Parameter("state", this, ParameterType.OTHER);

    public MxGRU(
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs) {
        this.opName = "RNN";
        this.stateSize = stateSize;
        this.dropRate = dropRate;
        this.numStackedLayers = numStackedLayers;
        this.useSequenceLength = useSequenceLength;
        this.useBidirectional = useBidirectional;
        this.stateOutputs = stateOutputs;
        this.mode = "gru";
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        Shape inputShape = inputs[0];
        return new Shape(inputShape.get(0), inputShape.get(1), stateSize);
    }

    @Override
    public List<Parameter> getDirectParameters() {
        List<Parameter> directParameters = new ArrayList<>(this.parameters);
        directParameters.add(state);
        return directParameters;
    }

    @Override
    public void beforeInitialize(NDList inputs) {
        NDArray input = inputs.head();
        Shape inputShape = input.getShape();
        if (!isLayoutSupported(EXPECTED_LAYOUT, inputShape.getLayout())) {
            throw new UnsupportedOperationException("RNN requires TNC layout");
        }
    }

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        NDArray input = inputs.get(0);
        long channelSize = input.getShape().get(2);
        long batchSize = input.getShape().get(1);
        switch (name) {
            case "i2rWeight":
            case "i2zWeight":
            case "i2nWeight":
                return new Shape(stateSize, channelSize);
            case "h2rWeight":
            case "h2zWeight":
            case "h2nWeight":
                return new Shape(stateSize, stateSize);
            case "h2rBias":
            case "i2rBias":
            case "h2zBias":
            case "i2zBias":
            case "h2nBias":
            case "i2nBias":
                return new Shape(stateSize);
            case "state":
                return new Shape(numStackedLayers, batchSize, stateSize);
            default:
                throw new IllegalArgumentException("Invalid parameter name: " + name);
        }
    }

    @Override
    protected NDList opInputs(NDList inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("RNN requires exactly 1 NDArray");
        }

        NDList result = new NDList();
        NDList parameterList = new NDList();
        for (Parameter parameter : parameters) {
            parameterList.add(parameter.getName(), parameter.getArray().flatten());
        }
        result.add(inputs.get(0));
        result.add(NDArrays.concat(parameterList));
        result.add(state.getArray());
        if (useSequenceLength) {
            result.add(inputs.get(1));
        }
        return result;
    }

    @Override
    protected PairList<String, Object> opParams(PairList<String, Object> params) {
        MxOpParams result = new MxOpParams();
        result.addParam("p", dropRate);
        result.addParam("state_size", stateSize);
        result.addParam("num_layers", numStackedLayers);
        result.addParam("use_sequence_length", useSequenceLength);
        result.addParam("bidirectional", useBidirectional);
        result.addParam("state_outputs", stateOutputs);
        result.addParam("mode", mode);
        result.addAll(params);
        return result;
    }
}
