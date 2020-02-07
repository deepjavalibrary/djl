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
package ai.djl.nn.recurrent;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are implemented,
 * with both multi-layer and bidirectional support.
 */
public abstract class RecurrentCell extends ParameterBlock {
    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.TIME, LayoutType.CHANNEL
    };

    protected long stateSize;
    protected float dropRate;
    protected int numStackedLayers;
    protected String mode;
    protected boolean useSequenceLength;
    protected boolean useBidirectional;
    protected int gates;
    protected byte currentVersion = 1;
    protected boolean stateOutputs;

    protected Shape stateShape;
    protected List<Parameter> parameters = new ArrayList<>();

    /**
     * Creates a {@code RecurrentCell} object.
     *
     * @param builder the {@code Builder} that has the necessary configurations
     */
    public RecurrentCell(BaseBuilder<?> builder) {
        stateSize = builder.stateSize;
        dropRate = builder.dropRate;
        numStackedLayers = builder.numStackedLayers;
        useSequenceLength = builder.useSequenceLength;
        useBidirectional = builder.useBidirectional;
        stateOutputs = builder.stateOutputs;

        for (int i = 0; i < numStackedLayers; i++) {
            // Preserve this order of parameters. It is important to maintain this order as we
            // concat the parameters.
            parameters.add(
                    new Parameter(String.format("l%d_i2h_weight", i), this, ParameterType.WEIGHT));
            parameters.add(
                    new Parameter(String.format("l%d_h2h_weight", i), this, ParameterType.WEIGHT));
            parameters.add(
                    new Parameter(String.format("l%d_i2h_bias", i), this, ParameterType.BIAS));
            parameters.add(
                    new Parameter(String.format("l%d_h2h_bias", i), this, ParameterType.BIAS));
            if (useBidirectional) {
                parameters.add(
                        new Parameter(
                                String.format("r%d_i2h_weight", i), this, ParameterType.WEIGHT));
                parameters.add(
                        new Parameter(
                                String.format("r%d_h2h_weight", i), this, ParameterType.WEIGHT));
                parameters.add(
                        new Parameter(String.format("r%d_i2h_bias", i), this, ParameterType.BIAS));
                parameters.add(
                        new Parameter(String.format("r%d_h2h_bias", i), this, ParameterType.BIAS));
            }
        }
    }

    protected void validateInputSize(NDList inputs) {
        int numberofInputsRequired = 1;
        if (useSequenceLength) {
            numberofInputsRequired = 2;
        }
        if (inputs.size() != numberofInputsRequired) {
            throw new IllegalArgumentException(
                    "Invalid number of inputs for RNN. Size of input NDList must be "
                            + numberofInputsRequired
                            + " when useSequenceLength is "
                            + useSequenceLength);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(parameterStore, inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        NDList output =
                ex.rnn(
                        inputs,
                        mode,
                        stateSize,
                        dropRate,
                        numStackedLayers,
                        useSequenceLength,
                        useBidirectional,
                        stateOutputs,
                        params);

        NDList result = new NDList(output.head().transpose(1, 0, 2));
        if (stateOutputs) {
            result.add(output.get(1));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        // Input shape at this point is TNC. Output Shape should be NTS
        Shape inputShape = inputs[0];
        return new Shape[] {new Shape(inputShape.get(1), inputShape.get(0), stateSize)};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return parameters;
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(Shape[] inputs) {
        this.inputShapes = inputs;
        Shape inputShape = inputs[0];
        Block.validateLayout(EXPECTED_LAYOUT, inputShape.getLayout());
        long batchSize = inputShape.get(0);
        inputs[0] = new Shape(inputShape.get(1), inputShape.get(0), inputShape.get(2));
        stateShape = new Shape(numStackedLayers, batchSize, stateSize);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        Shape shape = inputShapes[0];
        long inputSize = shape.get(2);
        if (name.contains("bias")) {
            return new Shape(gates * stateSize);
        }
        if (name.contains("i2h")) {
            return new Shape(gates * stateSize, inputSize);
        }
        if (name.contains("h2h")) {
            return new Shape(gates * stateSize, stateSize);
        }
        throw new IllegalArgumentException("Invalid parameter name");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(currentVersion);
        for (Parameter parameter : parameters) {
            parameter.save(os);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version != this.currentVersion) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        for (Parameter parameter : parameters) {
            parameter.load(manager, is);
        }
    }

    protected NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        validateInputSize(inputs);
        inputs = updateInputLayoutToTNC(inputs);
        NDArray head = inputs.head();
        Device device = head.getDevice();

        NDList result = new NDList(head);
        try (NDList parameterList = new NDList()) {
            for (Parameter parameter : parameters) {
                NDArray array = parameterStore.getValue(parameter, device);
                parameterList.add(array.flatten());
            }
            NDArray array = NDArrays.concat(parameterList);
            result.add(array);
        }
        result.add(inputs.head().getManager().zeros(stateShape));
        if (useSequenceLength) {
            result.add(inputs.get(1));
        }
        return result;
    }

    protected NDList updateInputLayoutToTNC(NDList inputs) {
        return new NDList(inputs.singletonOrThrow().transpose(1, 0, 2));
    }

    /** The Builder to construct a {@link RecurrentCell} type of {@link ai.djl.nn.Block}. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected float dropRate;
        protected long stateSize = -1;
        protected int numStackedLayers = -1;
        protected double lstmStateClipMin;
        protected double lstmStateClipMax;
        protected boolean clipLstmState;
        protected boolean useSequenceLength;
        protected boolean useBidirectional;
        protected boolean stateOutputs;
        protected RNN.Activation activation;

        /**
         * Sets the drop rate of the dropout on the outputs of each RNN layer, except the last
         * layer.
         *
         * @param dropRate the drop rate of the dropout
         * @return this Builder
         */
        public T optDropRate(float dropRate) {
            this.dropRate = dropRate;
            return self();
        }

        /**
         * Sets the <b>Required</b> size of the state for each layer.
         *
         * @param stateSize the size of the state for each layer
         * @return this Builder
         */
        public T setStateSize(int stateSize) {
            this.stateSize = stateSize;
            return self();
        }

        /**
         * Sets the <b>Required</b> number of stacked layers.
         *
         * @param numStackedLayers the number of stacked layers
         * @return this Builder
         */
        public T setNumStackedLayers(int numStackedLayers) {
            this.numStackedLayers = numStackedLayers;
            return self();
        }

        /**
         * Sets the optional parameter that indicates whether to include an extra input parameter
         * sequence_length to specify variable length sequence.
         *
         * @param useSequenceLength whether to use sequence length
         * @return this Builder
         */
        public T setSequenceLength(boolean useSequenceLength) {
            this.useSequenceLength = useSequenceLength;
            return self();
        }

        /**
         * Sets the optional parameter that indicates whether to use bidirectional recurrent layers.
         *
         * @param useBidirectional whether to use bidirectional recurrent layers
         * @return this Builder
         */
        public T optBidrectional(boolean useBidirectional) {
            this.useBidirectional = useBidirectional;
            return self();
        }

        /**
         * Sets the optional parameter that indicates whether to have the states as symbol outputs.
         *
         * @param stateOutputs whether to have the states as symbol output
         * @return this Builder
         */
        public T optStateOutput(boolean stateOutputs) {
            this.stateOutputs = stateOutputs;
            return self();
        }

        protected abstract T self();
    }
}
