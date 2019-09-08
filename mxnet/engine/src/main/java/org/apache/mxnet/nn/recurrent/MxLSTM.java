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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.recurrent.LSTM;
import software.amazon.ai.util.PairList;

public class MxLSTM extends MxNNBlock implements LSTM {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.TIME, LayoutType.BATCH, LayoutType.CHANNEL
    };

    private static final byte VERSION = 1;

    private long stateSize;
    private float dropRate;
    private double lstmStateClipMin;
    private double lstmStateClipMax;
    private int numStackedLayers;
    private String mode;
    private boolean useSequenceLength;
    private boolean useBidirectional;
    private boolean stateOutputs;
    private boolean clipLstmState;

    private List<Parameter> parameters =
            Arrays.asList(
                    new Parameter("i2iWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2iBias", this, ParameterType.BIAS),
                    new Parameter("h2iWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2iBias", this, ParameterType.BIAS),
                    new Parameter("i2fWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2fBias", this, ParameterType.BIAS),
                    new Parameter("h2fWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2fBias", this, ParameterType.BIAS),
                    new Parameter("i2gWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2gBias", this, ParameterType.BIAS),
                    new Parameter("h2gWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2gBias", this, ParameterType.BIAS),
                    new Parameter("i2oWeight", this, ParameterType.WEIGHT),
                    new Parameter("i2oBias", this, ParameterType.BIAS),
                    new Parameter("h2oWeight", this, ParameterType.WEIGHT),
                    new Parameter("h2oBias", this, ParameterType.BIAS));

    private Parameter state = new Parameter("state", this, ParameterType.OTHER);
    private Parameter stateCell = new Parameter("state_cell", this, ParameterType.OTHER);

    public MxLSTM(NDManager manager, LSTM.Builder builder) {
        super(manager);
        opName = "RNN";
        stateSize = builder.getStateSize();
        dropRate = builder.getDropRate();
        numStackedLayers = builder.getNumStackedLayers();
        useSequenceLength = builder.isUseSequenceLength();
        useBidirectional = builder.isUseBidirectional();
        stateOutputs = builder.isStateOutputs();
        mode = "lstm";
        clipLstmState = builder.isClipLstmState();
        if (clipLstmState) {
            lstmStateClipMin = builder.getLstmStateClipMin();
            lstmStateClipMax = builder.getLstmStateClipMax();
        }
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
        directParameters.add(stateCell);
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
            case "i2iWeight":
            case "i2fWeight":
            case "i2gWeight":
            case "i2oWeight":
                return new Shape(stateSize, channelSize);
            case "h2iWeight":
            case "h2fWeight":
            case "h2gWeight":
            case "h2oWeight":
                return new Shape(stateSize, stateSize);
            case "h2iBias":
            case "i2iBias":
            case "h2fBias":
            case "i2fBias":
            case "h2gBias":
            case "i2gBias":
            case "h2oBias":
            case "i2oBias":
                return new Shape(stateSize);
            case "state":
            case "state_cell":
                return new Shape(numStackedLayers, batchSize, stateSize);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
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
        result.add(stateCell.getArray());
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
        if (clipLstmState) {
            result.addParam("lstm_state_clip_nan", true);
            result.addParam("lstm_state_clip_min", lstmStateClipMin);
            result.addParam("lstm_state_clip_max", lstmStateClipMax);
        }
        result.addAll(params);
        return result;
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        for (Parameter parameter : parameters) {
            parameter.save(os);
        }
        state.save(os);
        stateCell.save(os);
    }

    @Override
    public void loadParameters(DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        for (Parameter parameter : parameters) {
            parameter.load(is);
        }
        state.load(is);
        stateCell.load(is);
    }
}
