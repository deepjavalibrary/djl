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
 * A LinearCollection block applies \(m\) linear transformations \(Y_i = X_i W_i + b_i\) for each
 * \(i \in [1, \ldots, m]\) and \(m = \prod_{j=1}^t s_j\). \(t\) is typically 1, so compared to a
 * {@link Linear} block, the involved shapes have typically one split dimension added which
 * separates the different linear transformations from each other. Another difference to a {@link
 * Linear} block is that the weight is not transposed (to align with the internally used algebraic
 * operation {@link NDArray#matMul(NDArray)} ). There is currently only a single batch dimension
 * \(x_1\) supported.
 *
 * <p>It has the following shapes:
 *
 * <ul>
 *   <li>input X: [x_1, s_1, s_2, …, s_t, input_dim]
 *   <li>weight W: [s_1, s_2, …, s_t, input_dim, units]
 *   <li>Bias b: [s_1, s_2, …, s_t, units]
 *   <li>output Y: [x_1, s_1, s_2, …, s_t, units]
 * </ul>
 *
 * <p>The LinearCollection block should be constructed using {@link LinearCollection.Builder}.
 */
public class LinearCollection extends AbstractBlock {

    private static final byte VERSION = 1;

    private long units;
    private long inputFeatures;

    private Shape inputShape;

    private Parameter weight;
    private Parameter bias;

    private int[] shiftBatchAxis;

    private int[] reverseShiftBatchAxis;

    LinearCollection(Builder builder) {
        super(VERSION);
        units = builder.units;
        weight =
                addParameter(
                        Parameter.builder()
                                .setName("weight")
                                .setType(Parameter.Type.WEIGHT)
                                .build());
        if (builder.bias) {
            bias =
                    addParameter(
                            Parameter.builder()
                                    .setName("bias")
                                    .setType(Parameter.Type.BIAS)
                                    .build());
        }
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
        NDArray biasArr = parameterStore.getValue(bias, device, training);
        return linear(input, weightArr, biasArr);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[] {inputs[0].slice(0, inputs[0].dimension() - 1).add(units)};
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
        inputFeatures = input.slice(1, input.dimension()).size();
        inputShape = input.slice(0, 1);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Shape[] inputShapes) {
        Shape input = inputShapes[0];
        weight.setShape(input.slice(1, input.dimension()).add(units));
        if (bias != null) {
            bias.setShape(input.slice(1, input.dimension() - 1).add(units));
        }
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
        switch (loadVersion) {
            case VERSION:
                units = is.readLong();
                inputFeatures = is.readLong();
                break;
            default:
                throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        inputShape = Shape.decode(is);
    }

    /**
     * Applies linear transformations to the incoming data.
     *
     * @param input input X: [x_1, s_1, s_2, …, s_t, input_dim]
     * @param weight weight W: [s_1, s_2, …, s_t, input_dim, units]
     * @param bias bias b: [s_1, s_2, …, s_t, units]
     * @return output Y: [x_1, s_1, s_2, …, s_t, units]
     */
    public NDList linear(NDArray input, NDArray weight, NDArray bias) {
        if (shiftBatchAxis == null) {
            // as the batch axis is the first axis in the shape of the input resp. output
            // arrays, it needs to be shifted in order to bring the split axes in front resp. back
            // again to be suitable for matMul;
            // in case there is only one split axis, the transpose array (1,0,2) could be used for
            // both shifts, but for the general case we calculate the transpose arrays here
            int dim = input.getShape().dimension();
            shiftBatchAxis = new int[dim];
            reverseShiftBatchAxis = new int[dim];
            for (int d = 0; d < dim - 2; d++) {
                shiftBatchAxis[d] = d + 1;
                reverseShiftBatchAxis[d + 1] = d;
            }
            shiftBatchAxis[dim - 1] = dim - 1;
            reverseShiftBatchAxis[dim - 1] = dim - 1;
            shiftBatchAxis[dim - 2] = 0;
            reverseShiftBatchAxis[0] = dim - 2;
        }
        NDArray resultArr =
                input.transpose(shiftBatchAxis).matMul(weight).transpose(reverseShiftBatchAxis);
        if (bias != null) {
            resultArr.addi(bias);
        }
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

    /** The Builder to construct a {@link LinearCollection} type of {@link Block}. */
    public static final class Builder {

        private long units;
        private boolean bias = true;

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
         * Sets the optional parameter that indicates whether to include a bias vector with default
         * value of true.
         *
         * @param bias whether to use a bias vector parameter
         * @return this Builder
         */
        public Builder optBias(boolean bias) {
            this.bias = bias;
            return this;
        }

        /**
         * Returns the constructed {@code Linear}.
         *
         * @return the constructed {@code Linear}
         * @throws IllegalArgumentException if all required parameters (outChannels) have not been
         *     set
         */
        public LinearCollection build() {
            Preconditions.checkArgument(units > 0, "You must specify unit");
            return new LinearCollection(this);
        }
    }
}
