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

import ai.djl.MalformedModelException;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.util.Pair;
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
    protected int numLayers;
    protected int gates;
    protected boolean batchFirst;
    protected boolean hasBiases;
    protected boolean bidirectional;
    protected boolean returnState;

    /**
     * Creates a {@code RecurrentBlock} object.
     *
     * @param builder the {@code Builder} that has the necessary configurations
     */
    public RecurrentBlock(BaseBuilder<?> builder) {
        super(VERSION);
        stateSize = builder.stateSize;
        dropRate = builder.dropRate;
        numLayers = builder.numLayers;
        batchFirst = builder.batchFirst;
        hasBiases = builder.hasBiases;
        bidirectional = builder.bidirectional;
        returnState = builder.returnState;

        Parameter.Type[] parameterTypes = {Parameter.Type.WEIGHT, Parameter.Type.BIAS};
        String[] directions = {"l"};
        if (builder.bidirectional) {
            directions = new String[] {"l", "r"};
        }
        String[] gateStrings = {"i2h", "h2h"};

        for (int i = 0; i < numLayers; i++) {
            for (Parameter.Type parameterType : parameterTypes) {
                for (String direction : directions) {
                    for (String gateString : gateStrings) {
                        String name =
                                direction + '_' + i + '_' + gateString + '_' + parameterType.name();
                        addParameter(
                                Parameter.builder().setName(name).setType(parameterType).build());
                    }
                }
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Shape inputShape = inputs[0];
        Shape outputShape =
                new Shape(inputShape.get(0), inputShape.get(1), stateSize * getNumDirections());
        if (!returnState) {
            return new Shape[] {
                outputShape,
            };
        }
        return new Shape[] {
            outputShape,
            new Shape(
                    (long) numLayers * getNumDirections(),
                    inputShape.get((batchFirst) ? 0 : 1),
                    stateSize)
        };
    }

    /** {@inheritDoc} */
    @Override
    protected void beforeInitialize(Shape... inputShapes) {
        super.beforeInitialize(inputShapes);
        Block.validateLayout(EXPECTED_LAYOUT, inputShapes[0].getLayout());
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Shape[] inputs) {
        Shape inputShape = inputs[0];
        ParameterList parameters = getDirectParameters();
        for (Pair<String, Parameter> pair : parameters) {
            String name = pair.getKey();
            Parameter parameter = pair.getValue();
            int layer = Integer.parseInt(name.split("_")[1]);
            long inputSize = inputShape.get(2);
            if (layer > 0) {
                inputSize = stateSize * getNumDirections();
            }
            if (name.contains("BIAS")) {
                parameter.setShape(new Shape(gates * stateSize));
            } else if (name.contains("i2h")) {
                parameter.setShape(new Shape(gates * stateSize, inputSize));
            } else if (name.contains("h2h")) {
                parameter.setShape(new Shape(gates * stateSize, stateSize));
            } else {
                throw new IllegalArgumentException("Invalid parameter name");
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == version) {
            readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }

    protected int getNumDirections() {
        return bidirectional ? 2 : 1;
    }

    /** The Builder to construct a {@link RecurrentBlock} type of {@link ai.djl.nn.Block}. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected float dropRate;
        protected long stateSize;
        protected int numLayers;
        // set it true by default for usability
        protected boolean batchFirst = true;
        protected boolean hasBiases = true;
        protected boolean bidirectional;
        protected boolean returnState;
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
         * @param numLayers the number of stacked layers
         * @return this Builder
         */
        public T setNumLayers(int numLayers) {
            this.numLayers = numLayers;
            return self();
        }

        /**
         * Sets the optional parameter that indicates whether to use bidirectional recurrent layers.
         *
         * @param useBidirectional whether to use bidirectional recurrent layers
         * @return this Builder
         */
        public T optBidirectional(boolean useBidirectional) {
            this.bidirectional = useBidirectional;
            return self();
        }

        /**
         * Sets the optional batchFirst flag that indicates whether the input is batch major or not.
         * The default value is true.
         *
         * @param batchFirst whether the input is batch major or not
         * @return this Builder
         */
        public T optBatchFirst(boolean batchFirst) {
            this.batchFirst = batchFirst;
            return self();
        }

        /**
         * Sets the optional biases flag that indicates whether to use biases or not.
         *
         * @param hasBiases whether to use biases or not
         * @return this Builder
         */
        public T optHasBiases(boolean hasBiases) {
            this.hasBiases = hasBiases;
            return self();
        }

        /**
         * Sets the optional flag that indicates whether to return state or not. This is typically
         * useful when you use RecurrentBlock in Sequential block. The default value is false.
         *
         * @param returnState whether to return state or not
         * @return this Builder
         */
        public T optReturnState(boolean returnState) {
            this.returnState = returnState;
            return self();
        }

        protected abstract T self();
    }
}
