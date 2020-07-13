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
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;

/**
 * {@code RecurrentBlock} is an abstract implementation of recurrent neural networks.
 *
 * <p>Recurrent neural networks are neural networks with hidden states. They are very popular for
 * natural language processing tasks, and other tasks which involve sequential data.
 *
 * <p>This [article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) written by Andrej
 * Karpathy provides a detailed explanation of recurrent neural networks.
 *
 * <p>Currently, vanilla RNN, LSTM and GRU are implemented, with both multi-layer and bidirectional
 * support.
 */
public abstract class RecurrentBlock extends AbstractBlock {

    private static final byte VERSION = 2;

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.TIME, LayoutType.CHANNEL
    };

    protected long stateSize;
    protected float dropRate;
    protected int numStackedLayers;
    protected String mode;
    protected boolean useSequenceLength;
    protected int numDirections = 1;
    protected int gates;
    protected boolean stateOutputs;
    protected NDArray beginState;

    /**
     * Creates a {@code RecurrentBlock} object.
     *
     * @param builder the {@code Builder} that has the necessary configurations
     */
    public RecurrentBlock(BaseBuilder<?> builder) {
        super(VERSION);
        stateSize = builder.stateSize;
        dropRate = builder.dropRate;
        numStackedLayers = builder.numStackedLayers;
        useSequenceLength = builder.useSequenceLength;
        stateOutputs = builder.stateOutputs;
        if (builder.useBidirectional) {
            numDirections = 2;
        }

        ParameterType[] parameterTypes = {ParameterType.WEIGHT, ParameterType.BIAS};
        String[] directions = {"l"};
        if (builder.useBidirectional) {
            directions = new String[] {"l", "r"};
        }
        String[] gateStrings = {"i2h", "h2h"};

        for (ParameterType parameterType : parameterTypes) {
            for (int i = 0; i < numStackedLayers; i++) {
                for (String direction : directions) {
                    for (String gateString : gateStrings) {
                        addParameter(
                                new Parameter(
                                        String.format(
                                                "%s_%s_%s_%s",
                                                direction, i, gateString, parameterType.name()),
                                        this,
                                        parameterType));
                    }
                }
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

    /**
     * Sets the parameter that indicates whether the output must include the hidden states.
     *
     * @param stateOutputs whether the output must include the hidden states.
     */
    public final void setStateOutputs(boolean stateOutputs) {
        this.stateOutputs = stateOutputs;
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
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
                        isBidirectional(),
                        true,
                        params);

        NDList result = new NDList(output.head().transpose(1, 0, 2));
        if (stateOutputs) {
            result.add(output.get(1));
        }
        resetBeginStates();
        return result;
    }

    /**
     * Sets the initial {@link NDArray} value for the hidden states.
     *
     * @param beginStates the {@link NDArray} value for the hidden states
     */
    public void setBeginStates(NDList beginStates) {
        this.beginState = beginStates.get(0);
    }

    protected void resetBeginStates() {
        beginState = null;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        // Input shape at this point is TNC. Output Shape should be NTS
        Shape inputShape = inputs[0];
        if (stateOutputs) {
            return new Shape[] {
                new Shape(inputShape.get(1), inputShape.get(0), stateSize * numDirections),
                new Shape(numStackedLayers * numDirections, inputShape.get(1), stateSize)
            };
        }
        return new Shape[] {
            new Shape(inputShape.get(1), inputShape.get(0), stateSize * numDirections)
        };
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(Shape[] inputs) {
        this.inputShapes = inputs;
        Shape inputShape = inputs[0];
        Block.validateLayout(EXPECTED_LAYOUT, inputShape.getLayout());
        inputs[0] = new Shape(inputShape.get(1), inputShape.get(0), inputShape.get(2));
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        int layer = Integer.parseInt(name.split("_")[1]);
        Shape shape = inputShapes[0];
        long inputs = shape.get(2);
        if (layer > 0) {
            inputs = stateSize * numDirections;
        }
        if (name.contains("BIAS")) {
            return new Shape(gates * stateSize);
        }
        if (name.contains("i2h")) {
            return new Shape(gates * stateSize, inputs);
        }
        if (name.contains("h2h")) {
            return new Shape(gates * stateSize, stateSize);
        }
        throw new IllegalArgumentException("Invalid parameter name");
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte version, DataInputStream is)
            throws IOException, MalformedModelException {
        if (version == VERSION) {
            readInputShapes(is);
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
    }

    protected boolean isBidirectional() {
        return numDirections == 2;
    }

    protected NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        validateInputSize(inputs);
        long batchSize = inputs.head().getShape().get(0);
        inputs = updateInputLayoutToTNC(inputs);
        NDArray head = inputs.head();
        Device device = head.getDevice();

        NDList result = new NDList(head);
        try (NDList parameterList = new NDList()) {
            for (Parameter parameter : parameters.values()) {
                NDArray array = parameterStore.getValue(parameter, device);
                parameterList.add(array.flatten());
            }
            NDArray array = NDArrays.concat(parameterList);
            result.add(array);
        }
        Shape stateShape = new Shape(numStackedLayers * numDirections, batchSize, stateSize);
        if (beginState != null) {
            result.add(beginState);
        } else {
            result.add(inputs.head().getManager().zeros(stateShape));
        }
        if (useSequenceLength) {
            result.add(inputs.get(1));
        }
        return result;
    }

    protected NDList updateInputLayoutToTNC(NDList inputs) {
        return new NDList(inputs.singletonOrThrow().transpose(1, 0, 2));
    }

    /** The Builder to construct a {@link RecurrentBlock} type of {@link ai.djl.nn.Block}. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected float dropRate;
        protected long stateSize;
        protected int numStackedLayers;
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
