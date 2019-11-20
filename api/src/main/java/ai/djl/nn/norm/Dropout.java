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
package ai.djl.nn.norm;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * Applies dropout operation to an input array.
 *
 * <p>During training, each element of the input is set to zero with probability p. The whole array
 * is rescaled by 1/(1−p) to keep the expected sum of the input unchanged. During testing, this
 * operator does not change the input if mode is ‘training’. If mode is ‘always’, the same
 * computation as during training will be applied.
 */
public class Dropout extends ParameterBlock {

    private static final byte VERSION = 1;

    private float probability;
    private int[] sharedAxes;

    Dropout(Builder builder) {
        probability = builder.probability;
        sharedAxes = builder.sharedAxes;
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        NDArrayEx ex = inputs.singletonOrThrow().getNDArrayInternal();
        return ex.dropout(inputs, probability, sharedAxes, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[] {inputShapes[0]};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("Dropout has no parameters");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
    }

    /** The Builder to construct a {@link Dropout} type of {@link ai.djl.nn.Block}. */
    public static final class Builder {

        private float probability = 0.5f;
        private int[] sharedAxes = {};

        /**
         * Sets the probability or the fraction of the input that gets dropped out during training
         * time. Defaults to 0.5.
         *
         * @param probability fraction of the input that gets dropped out during training
         * @return this Builder
         */
        public Builder optProbability(float probability) {
            this.probability = probability;
            return this;
        }

        /**
         * Sets the axes for variational dropout kernel.
         *
         * @param sharedAxes the axes for variational dropout kernel
         * @return this Builder
         */
        public Builder optSharedAxes(int[] sharedAxes) {
            this.sharedAxes = sharedAxes;
            return this;
        }

        /**
         * Builds a {@link Dropout} block.
         *
         * @return the {@link Dropout} block
         */
        public Dropout build() {
            return new Dropout(this);
        }
    }
}
