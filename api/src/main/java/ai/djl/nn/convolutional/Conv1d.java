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
package ai.djl.nn.convolutional;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.util.Preconditions;

/**
 * A {@code Conv1d} layer works similar to {@link Convolution} layer with the exception of the
 * number of dimension it operates on being only one, which is {@link LayoutType#WIDTH}. The channel
 * of the input data may be more than one, depending on what data is processed. Each filter slides
 * through the data with only one direction of movement along the dimension itself.
 *
 * <p>Commonly, this kind of convolution layer, as proposed in this <a
 * href="https://ieeexplore.ieee.org/document/7318926/">paper</a> is used in tasks utilizing serial
 * data, enabling convolutional processing of 1-dimensional data such as time-series data (stock
 * price, weather, ECG) and text/speech data without the need of transforming it to 2-dimensional
 * data to be processed by {@link Conv2d}, though this is quite a common technique as well.
 *
 * <p>The input to a {@code Conv1d} is an {@link ai.djl.ndarray.NDList} with a single 3-D {@link
 * ai.djl.ndarray.NDArray}. The layout of the {@link ai.djl.ndarray.NDArray} must be "NCW". The
 * shapes are
 *
 * <ul>
 *   <li>{@code data: (batch_size, channel, width)}
 *   <li>{@code weight: (num_filter, channel, kernel[0])}
 *   <li>{@code bias: (num_filter,)}
 *   <li>{@code out: (batch_size, num_filter, out_width)} <br>
 *       {@code out_width = f(width, kernel[0], pad[0], stride[0], dilate[0])} <br>
 *       {@code where f(x, k, p, s, d) = floor((x + 2 * p - d * (k - 1) - 1)/s) + 1}
 * </ul>
 *
 * <p>Both {@code weight} and {@code bias} are learn-able parameters.
 *
 * @see Convolution
 */
public class Conv1d extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.WIDTH
    };

    private static final String STRING_LAYOUT = "NCW";
    private static final int NUM_DIMENSIONS = 3;

    Conv1d(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    protected LayoutType[] getExpectedLayout() {
        return EXPECTED_LAYOUT;
    }

    /** {@inheritDoc} */
    @Override
    protected String getStringLayout() {
        return STRING_LAYOUT;
    }

    /** {@inheritDoc} */
    @Override
    protected int numDimensions() {
        return NUM_DIMENSIONS;
    }

    /**
     * Applies 1D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @return the output of the conv1d operation
     */
    public static NDList conv1d(NDArray input, NDArray weight) {
        return conv1d(input, weight, null, new Shape(1), new Shape(0), new Shape(1));
    }

    /**
     * Applies 1D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @return the output of the conv1d operation
     */
    public static NDList conv1d(NDArray input, NDArray weight, NDArray bias) {
        return conv1d(input, weight, bias, new Shape(1), new Shape(0), new Shape(1));
    }

    /**
     * Applies 1D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(width)
     * @return the output of the conv1d operation
     */
    public static NDList conv1d(NDArray input, NDArray weight, NDArray bias, Shape stride) {
        return conv1d(input, weight, bias, stride, new Shape(0), new Shape(1));
    }

    /**
     * Applies 1D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(width)
     * @param padding implicit paddings on both sides of the input: Shape(width)
     * @return the output of the conv1d operation
     */
    public static NDList conv1d(
            NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding) {
        return conv1d(input, weight, bias, stride, padding, new Shape(1));
    }

    /**
     * Applies 1D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(width)
     * @param padding implicit paddings on both sides of the input: Shape(width)
     * @param dilation the spacing between kernel elements: Shape(width)
     * @return the output of the conv1d operation
     */
    public static NDList conv1d(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation) {
        return conv1d(input, weight, bias, stride, padding, dilation, 1);
    }

    /**
     * Applies 1D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(width)
     * @param padding implicit paddings on both sides of the input: Shape(width)
     * @param dilation the spacing between kernel elements: Shape(width)
     * @param groups split input into groups: input channel(input.size(1)) should be divisible by
     *     the number of groups
     * @return the output of the conv1d operation
     */
    public static NDList conv1d(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 3 && weight.getShape().dimension() == 3,
                "the shape of input or weight doesn't match the conv1d");
        Preconditions.checkArgument(
                stride.dimension() == 1 && padding.dimension() == 1 && dilation.dimension() == 1,
                "the shape of stride or padding or dilation doesn't match the conv1d");
        return Convolution.convolution(input, weight, bias, stride, padding, dilation, groups);
    }

    /**
     * Creates a builder to build a {@code Conv1d}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Conv1d} type of {@link Block}. */
    public static final class Builder extends ConvolutionBuilder<Builder> {

        /** Creates a builder that can build a {@link Conv1d} block. */
        Builder() {
            stride = new Shape(1);
            padding = new Shape(0);
            dilation = new Shape(1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Conv1d} block.
         *
         * @return the {@link Conv1d} block
         */
        public Conv1d build() {
            validate();
            return new Conv1d(this);
        }
    }
}
