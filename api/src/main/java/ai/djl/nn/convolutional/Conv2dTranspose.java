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
 * The input to a {@code Conv2dTranspose} is an {@link ai.djl.ndarray.NDList} with a single 4-D
 * {@link ai.djl.ndarray.NDArray}. The layout of the {@link ai.djl.ndarray.NDArray} must be "NCHW".
 * The shapes are
 *
 * <ul>
 *   <li>{@code data: (batch_size, channel, height, width)}
 *   <li>{@code weight: (num_filter, channel, kernel[0], kernel[1])}
 *   <li>{@code bias: (num_filter,)}
 *   <li>{@code out: (batch_size, num_filter, out_height, out_width)} <br>
 *       {@code out_height = f(height, kernel[0], pad[0], oPad[0], stride[0], dilate[0])} <br>
 *       {@code out_width = f(width, kernel[1], pad[1], oPad[1], stride[1], dilate[1])} <br>
 *       {@code where f(x, k, p, oP, s, d) = (x-1)*s-2*p+k+oP}
 * </ul>
 *
 * <p>Both {@code weight} and {@code bias} are learn-able parameters.
 *
 * @see Deconvolution
 */
public class Conv2dTranspose extends Deconvolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.HEIGHT, LayoutType.WIDTH
    };

    private static final String STRING_LAYOUT = "NCHW";
    private static final int NUM_DIMENSIONS = 4;

    Conv2dTranspose(Builder builder) {
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
     * Applies 2D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @return the output of the conv2dTranspose operation
     */
    public static NDList conv2dTranspose(NDArray input, NDArray weight) {
        return conv2dTranspose(
                input,
                weight,
                null,
                new Shape(1, 1),
                new Shape(0, 0),
                new Shape(0, 0),
                new Shape(1, 1));
    }

    /**
     * Applies 2D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @return the output of the conv2dTranspose operation
     */
    public static NDList conv2dTranspose(NDArray input, NDArray weight, NDArray bias) {
        return conv2dTranspose(
                input,
                weight,
                bias,
                new Shape(1, 1),
                new Shape(0, 0),
                new Shape(0, 0),
                new Shape(1, 1));
    }

    /**
     * Applies 2D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(height, width)
     * @return the output of the conv2dTranspose operation
     */
    public static NDList conv2dTranspose(
            NDArray input, NDArray weight, NDArray bias, Shape stride) {
        return conv2dTranspose(
                input, weight, bias, stride, new Shape(0, 0), new Shape(0, 0), new Shape(1, 1));
    }

    /**
     * Applies 2D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(height, width)
     * @param padding implicit paddings on both sides of the input: Shape(height, width)
     * @return the output of the conv2dTranspose operation
     */
    public static NDList conv2dTranspose(
            NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding) {
        return conv2dTranspose(
                input, weight, bias, stride, padding, new Shape(0, 0), new Shape(1, 1));
    }

    /**
     * Applies 2D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(height, width)
     * @param padding implicit paddings on both sides of the input: Shape(height, width)
     * @param outPadding Controls the amount of implicit zero-paddings on both sides of the output
     *     for outputPadding number of points for each dimension.
     * @return the output of the conv2dTranspose operation
     */
    public static NDList conv2dTranspose(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding) {
        return conv2dTranspose(input, weight, bias, stride, padding, outPadding, new Shape(1, 1));
    }

    /**
     * Applies 2D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(height, width)
     * @param padding implicit paddings on both sides of the input: Shape(height, width)
     * @param outPadding Controls the amount of implicit zero-paddings on both sides of the output
     *     for outputPadding number of points for each dimension.
     * @param dilation the spacing between kernel elements: Shape(height, width)
     * @return the output of the conv2dTranspose operation
     */
    public static NDList conv2dTranspose(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation) {
        return conv2dTranspose(input, weight, bias, stride, padding, outPadding, dilation, 1);
    }

    /**
     * Applies 2D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(height, width)
     * @param padding implicit paddings on both sides of the input: Shape(height, width)
     * @param outPadding Controls the amount of implicit zero-paddings on both sides of the output
     *     for outputPadding number of points for each dimension. Shape(height, width)
     * @param dilation the spacing between kernel elements: Shape(height, width)
     * @param groups split input into groups: input channel(input.size(1)) should be divisible by
     *     the number of groups
     * @return the output of the conv2dTranspose operation
     */
    public static NDList conv2dTranspose(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 4 && weight.getShape().dimension() == 4,
                "the shape of input or weight doesn't match the conv2dTranspose");
        Preconditions.checkArgument(
                stride.dimension() == 2
                        && padding.dimension() == 2
                        && outPadding.dimension() == 2
                        && dilation.dimension() == 2,
                "the shape of stride or padding or dilation doesn't match the conv2dTranspose");
        return Deconvolution.deconvolution(
                input, weight, bias, stride, padding, outPadding, dilation, groups);
    }

    /**
     * Creates a builder to build a {@code Conv2dTranspose}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Conv2dTranspose} type of {@link Block}. */
    public static final class Builder extends DeconvolutionBuilder<Builder> {

        /** Creates a builder that can build a {@link Conv2dTranspose} block. */
        Builder() {
            stride = new Shape(1, 1);
            padding = new Shape(0, 0);
            outPadding = new Shape(0, 0);
            dilation = new Shape(1, 1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Conv2dTranspose} block.
         *
         * @return the {@link Conv2dTranspose} block
         */
        public Conv2dTranspose build() {
            validate();
            return new Conv2dTranspose(this);
        }
    }
}
