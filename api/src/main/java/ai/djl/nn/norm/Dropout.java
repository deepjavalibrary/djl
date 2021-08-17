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

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;

/**
 * A dropout layer benefits a network by allowing some units (neurons), and hence their respective
 * connections, of a network to be randomly and temporarily removed by setting its value to 0
 * <b>only</b> during training by specified probability \(p\), usually set to 0.5. The use of
 * dropout acts as if multiple networks with different architectures had been trained, and during
 * test/inference, the removed unit's output is multiplied by \(p\) as an approximation of the
 * averaged output of all the possible network architectures for that unit. The implementation of
 * dropout gives state-of-the-art performances for diverse tasks as shown in the proposal's <a
 * href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">paper</a>, suggesting its
 * general-use capability.
 *
 * <p>The idea of dropout itself was proposed in 2014, with the purpose of improving the performance
 * of large networks due to co-adaptation, where some connections are stronger and learned more
 * while other connections become weaker and loses their impact on the prediction, resulting in
 * network overfitting. It was also created as an alternative for costly networks, such as large or
 * ensemble networks, by removing several units, hence creating different thinned network
 * architectures and simulates multiple networks within a single network, greatly reducing the
 * computation cost.
 *
 * <p>Dropout is recommended to be used when one is trying to optimize an overfitting network or
 * when large dataset is available. It is still quite commonly used in many publications due to its
 * generalization capability. However, using dropout may not prevent overfitting due to variation
 * and limited size of the dataset, and it is reported that dropout layer increases training time by
 * 2-3 times since different simulated multiple networks are trained for each iteration, thus
 * resulting in noisy parameter updates.
 *
 * @see <a href="https://d2l.djl.ai/chapter_multilayer-perceptrons/dropout.html">The D2L chapter on
 *     dropout</a>
 */
public class Dropout extends AbstractBlock {

    private static final byte VERSION = 2;

    private float rate;

    Dropout(Builder builder) {
        super(VERSION);
        rate = builder.rate;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return dropout(inputs.singletonOrThrow(), rate, training);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {inputShapes[0]};
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

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "Dropout()";
    }

    /**
     * Applies Dropout to the input.
     *
     * @param input input to apply dropout
     * @return output
     */
    public static NDList dropout(NDArray input) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.dropout(input, 0.5f, true);
    }

    /**
     * Applies Dropout to the input.
     *
     * @param input input to apply dropout
     * @param rate Fraction of the input units to drop
     * @return output
     */
    public static NDList dropout(NDArray input, float rate) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.dropout(input, rate, true);
    }

    /**
     * Applies Dropout to the input.
     *
     * @param input input to apply dropout
     * @param rate Fraction of the input units to drop
     * @param training apply dropout if true
     * @return output
     */
    public static NDList dropout(NDArray input, float rate, boolean training) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.dropout(input, rate, training);
    }

    /**
     * Creates a builder to build a {@link Dropout}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Dropout} type of {@link ai.djl.nn.Block}. */
    public static final class Builder {

        private float rate = 0.5f;

        Builder() {}

        /**
         * Sets the probability or the fraction of the input that gets dropped out during training
         * time. Defaults to 0.5.
         *
         * @param rate fraction of the input that gets dropped out during training
         * @return this Builder
         */
        public Builder optRate(float rate) {
            this.rate = rate;
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
