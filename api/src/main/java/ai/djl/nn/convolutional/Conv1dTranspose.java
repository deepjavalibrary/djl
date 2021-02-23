/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
 * A {@code Conv1dTranspose} layer works similar to {@link Deconvolution} layer with the exception
 * of the number of dimension it operates on being only one, which is {@link LayoutType#WIDTH}. The
 * channel of the input data may be more than one, depending on what data is processed. Each filter
 * slides through the data with only one direction of movement along the dimension itself.
 *
 * <p>The input to a {@code Conv1dTranspose} is an {@link ai.djl.ndarray.NDList} with a single 3-D
 * {@link ai.djl.ndarray.NDArray}. The layout of the {@link ai.djl.ndarray.NDArray} must be "NCW".
 * The shapes are
 *
 * <ul>
 *   <li>{@code data: (batch_size, channel, width)}
 *   <li>{@code weight: (num_filter, channel, kernel[0])}
 *   <li>{@code bias: (num_filter,)}
 *   <li>{@code out: (batch_size, num_filter, out_width)} <br>
 *       {@code out_width = f(width, kernel[0], pad[0], oPad[0], stride[0], dilate[0])} <br>
 *       {@code where f(x, k, p, oP, s, d) = (x-1)*s-2*p+k+oP}
 * </ul>
 *
 * <p>Both {@code weight} and {@code bias} are learn-able parameters.
 *
 * @see Deconvolution
 */
public class Conv1dTranspose extends Deconvolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.WIDTH
    };

    private static final String STRING_LAYOUT = "NCW";
    private static final int NUM_DIMENSIONS = 3;

    Conv1dTranspose(Builder builder) {
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
     * Applies 1D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @return the output of the conv1dTranspose operation
     */
    public static NDList conv1dTranspose(NDArray input, NDArray weight) {
        return conv1dTranspose(
                input, weight, null, new Shape(1), new Shape(0), new Shape(0), new Shape(1));
    }

    /**
     * Applies 1D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @return the output of the conv1dTranspose operation
     */
    public static NDList conv1dTranspose(NDArray input, NDArray weight, NDArray bias) {
        return conv1dTranspose(
                input, weight, bias, new Shape(1), new Shape(0), new Shape(0), new Shape(1));
    }

    /**
     * Applies 1D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(width)
     * @return the output of the conv1dTranspose operation
     */
    public static NDList conv1dTranspose(
            NDArray input, NDArray weight, NDArray bias, Shape stride) {
        return conv1dTranspose(
                input, weight, bias, stride, new Shape(0), new Shape(0), new Shape(1));
    }

    /**
     * Applies 1D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(width)
     * @param padding implicit paddings on both sides of the input: Shape(width)
     * @return the output of the conv1dTranspose operation
     */
    public static NDList conv1dTranspose(
            NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding) {
        return conv1dTranspose(input, weight, bias, stride, padding, new Shape(0), new Shape(1));
    }

    /**
     * Applies 1D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(width)
     * @param padding implicit paddings on both sides of the input: Shape(width)
     * @param outPadding Controls the amount of implicit zero-paddings on both sides of the output
     *     for outputPadding number of points for each dimension.
     * @return the output of the conv1dTranspose operation
     */
    public static NDList conv1dTranspose(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding) {
        return conv1dTranspose(input, weight, bias, stride, padding, outPadding, new Shape(1));
    }

    /**
     * Applies 1D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(width)
     * @param padding implicit paddings on both sides of the input: Shape(width)
     * @param outPadding Controls the amount of implicit zero-paddings on both sides of the output
     *     for outputPadding number of points for each dimension.
     * @param dilation the spacing between kernel elements: Shape(width)
     * @return the output of the conv1dTranspose operation
     */
    public static NDList conv1dTranspose(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation) {
        return conv1dTranspose(input, weight, bias, stride, padding, outPadding, dilation, 1);
    }

    /**
     * Applies 1D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(width)
     * @param padding implicit paddings on both sides of the input: Shape(width)
     * @param outPadding Controls the amount of implicit zero-paddings on both sides of the output
     *     for outputPadding number of points for each dimension. Shape(width)
     * @param dilation the spacing between kernel elements: Shape(width)
     * @param groups split input into groups: input channel(input.size(1)) should be divisible by
     *     the number of groups
     * @return the output of the conv1dTranspose operation
     */
    public static NDList conv1dTranspose(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 3 && weight.getShape().dimension() == 3,
                "the shape of input or weight doesn't match the conv1dTranspose");
        Preconditions.checkArgument(
                stride.dimension() == 1
                        && padding.dimension() == 1
                        && outPadding.dimension() == 1
                        && dilation.dimension() == 1,
                "the shape of stride or padding or dilation doesn't match the conv1dTranspose");
        return Deconvolution.deconvolution(
                input, weight, bias, stride, padding, outPadding, dilation, groups);
    }

    /**
     * Creates a builder to build a {@code Conv1dTranspose}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Conv1dTranspose} type of {@link Block}. */
    public static final class Builder extends DeconvolutionBuilder<Builder> {

        /** Creates a builder that can build a {@link Conv1dTranspose} block. */
        Builder() {
            stride = new Shape(1);
            padding = new Shape(0);
            outPadding = new Shape(0);
            dilation = new Shape(1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Conv1dTranspose} block.
         *
         * @return the {@link Conv1dTranspose} block
         */
        public Conv1dTranspose build() {
            validate();
            return new Conv1dTranspose(this);
        }
    }
}
