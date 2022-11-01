/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn.core;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;

/**
 * A Multiplication block performs an element-wise multiplication of inputs and weights as opposed
 * to a {@link Linear} block which additionally sums up each element-wise multiplication.
 *
 * <p>Similar to a {@link LinearCollection}, multiple split dimensions are supported but they remain
 * optional (i.e. \(t\) can be zero). Other differences to a {@link Linear} block are that the
 * weight has an additional dimension of size 1 interspersed (to broadcast the weight to every input
 * of the batch when applying the internally used algebraic operation {@link NDArray#mul(NDArray)} )
 * and that biases are not supported.
 *
 * <p>Caution: the output-channel is the left-most dimension as opposed to traditionally being the
 * right-most dimension. As the output is one dimension larger than that of a {@link Linear} block,
 * it is more efficient and therefore recommended to apply an aggregating function (like the sum)
 * first and only then shift the first axis of the aggregated and thus smaller {@link NDArray}
 * instance into last position.
 *
 * <p>It has the following shapes:
 *
 * <ul>
 *   <li>input X: [x_1, s_1, s_2, …, s_t, input_dim]
 *   <li>weight W: [units, 1, s_1, s_2, …, s_t, input_dim]
 *   <li>output Y: [units, x_1, s_1, s_2, …, s_t, input_dim]
 * </ul>
 *
 * <p>The Multiplication block should be constructed using {@link Multiplication.Builder}.
 */
public class Multiplication extends AbstractBlock {

    private static final byte VERSION = 1;

    private long units;
    private long inputFeatures;

    private Shape inputShape;

    private Parameter weight;

    Multiplication(Builder builder) {
        super(VERSION);
        units = builder.units;
        weight =
                addParameter(
                        Parameter.builder()
                                .setName("weight")
                                .setType(Parameter.Type.WEIGHT)
                                .build());
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray weightArr = parameterStore.getValue(weight, device, training);
        return multiply(input, weightArr);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[] {new Shape(units).addAll(inputs[0])};
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        return new PairList<>(
                Collections.singletonList("linearInput"), Collections.singletonList(inputShape));
    }

    /** {@inheritDoc} */
    @Override
    protected void beforeInitialize(Shape... inputShapes) {
        super.beforeInitialize(inputShapes);
        Preconditions.checkArgument(inputShapes.length == 1, "Linear block only support 1 input");
        Shape input = inputShapes[0];
        inputFeatures = input.slice(1).size();
        inputShape = input.slice(0, 1);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Shape[] inputShapes) {
        Shape input = inputShapes[0];
        weight.setShape(new Shape(units, 1).addAll(input.slice(1)));
    }

    /** {@inheritDoc} */
    @Override
    protected void saveMetadata(DataOutputStream os) throws IOException {
        os.writeLong(units);
        os.writeLong(inputFeatures);
        os.write(inputShape.getEncoded());
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == VERSION) {
            units = is.readLong();
            inputFeatures = is.readLong();
        } else {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        inputShape = Shape.decode(is);
    }

    /**
     * Applies an element-wise multiplication to the incoming data.
     *
     * @param input The incoming data
     * @param weight The weight of this block
     * @return element-wise multiplication of input and weight using broadcasting rules
     */
    public NDList multiply(NDArray input, NDArray weight) {
        NDArray resultArr = input.mul(weight);
        return new NDList(resultArr);
    }

    /**
     * Creates a builder to build a {@code Linear}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Multiplication} type of {@link Block}. */
    public static final class Builder {

        private long units;

        Builder() {}

        /**
         * Sets the number of output channels.
         *
         * @param units the number of desired output channels
         * @return this Builder
         */
        public Builder setUnits(long units) {
            this.units = units;
            return this;
        }

        /**
         * Returns the constructed {@code Linear}.
         *
         * @return the constructed {@code Linear}
         * @throws IllegalArgumentException if all required parameters (outChannels) have not been
         *     set
         */
        public Multiplication build() {
            Preconditions.checkArgument(units > 0, "You must specify unit");
            return new Multiplication(this);
        }
    }
}
