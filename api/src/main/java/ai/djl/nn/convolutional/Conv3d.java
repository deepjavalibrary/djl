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
 * {@code Conv3d} layer behaves just as {@link Convolution} does, with the distinction being it
 * operates of 3-dimensional data such as medical images or video data. The traversal of each filter
 * begins from {@link LayoutType#WIDTH} then to {@link LayoutType#HEIGHT}, and lastly across each
 * {@link LayoutType#DEPTH} in the specified {@code depth} size of the data.
 *
 * <p>The utilization of {@code Conv3d} layer allows deeper analysis of visual data such as those in
 * medical images, or even analysis on temporal data such as video data as a whole instead of
 * processing each frame with a {@link Conv2d} layer, despite this being a common practice in
 * computer vision researches. The benefit of utilizing this kind of layer is the maintaining of
 * serial data across 2-dimensional data, hence could be beneficial for research focus on such as
 * object tracking. The drawback is that this kind of layer is more costly compared to other
 * convolution layer types since dot product operation is performed on all three dimensions.
 *
 * <p>The input to a {@code Conv3d} is an {@link ai.djl.ndarray.NDList} with a single 5-D {@link
 * ai.djl.ndarray.NDArray}. The layout of the {@link ai.djl.ndarray.NDArray} must be "NCDHW". The
 * shapes are
 *
 * <ul>
 *   <li>{@code data: (batch_size, channel, depth, height, width)}
 *   <li>{@code weight: (num_filter, channel, kernel[0], kernel[1], kernel[2])}
 *   <li>{@code bias: (num_filter,)}
 *   <li>{@code out: (batch_size, num_filter, out_depth, out_height, out_width)} <br>
 *       {@code out_depth = f(depth, kernel[0], pad[0], stride[0], dilate[0])} <br>
 *       {@code out_height = f(height, kernel[1], pad[1], stride[1], dilate[1])} <br>
 *       {@code out_width = f(width, kernel[2], pad[2], stride[2], dilate[2])} <br>
 *       {@code where f(x, k, p, s, d) = floor((x + 2 * p - d * (k - 1) - 1)/s) + 1}
 * </ul>
 *
 * <p>Both {@code weight} and {@code bias} are learn-able parameters.
 *
 * @see Convolution
 */
public class Conv3d extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.DEPTH, LayoutType.HEIGHT, LayoutType.WIDTH
    };
    private static final String STRING_LAYOUT = "NCDHW";
    private static final int NUM_DIMENSIONS = 5;

    Conv3d(Builder builder) {
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
     * Applies 3D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, depth, height,
     *     width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, depth,
     *     height, width)
     * @return the output of the conv3d operation
     */
    public static NDList conv3d(NDArray input, NDArray weight) {
        return conv3d(
                input, weight, null, new Shape(1, 1, 1), new Shape(0, 0, 0), new Shape(1, 1, 1));
    }

    /**
     * Applies 3D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, depth, height,
     *     width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, depth,
     *     height, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @return the output of the conv3d operation
     */
    public static NDList conv3d(NDArray input, NDArray weight, NDArray bias) {
        return conv3d(
                input, weight, bias, new Shape(1, 1, 1), new Shape(0, 0, 0), new Shape(1, 1, 1));
    }

    /**
     * Applies 3D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, depth, height,
     *     width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, depth,
     *     height, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(depth, height, width)
     * @return the output of the conv3d operation
     */
    public static NDList conv3d(NDArray input, NDArray weight, NDArray bias, Shape stride) {
        return conv3d(input, weight, bias, stride, new Shape(0, 0, 0), new Shape(1, 1, 1));
    }

    /**
     * Applies 3D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, depth, height,
     *     width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, depth,
     *     height, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(depth, height, width)
     * @param padding implicit paddings on both sides of the input: Shape(depth, height, width)
     * @return the output of the conv3d operation
     */
    public static NDList conv3d(
            NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding) {
        return conv3d(input, weight, bias, stride, padding, new Shape(1, 1, 1));
    }

    /**
     * Applies 3D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, depth, height,
     *     width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, depth,
     *     height, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(depth, height, width)
     * @param padding implicit paddings on both sides of the input: Shape(depth, height, width)
     * @param dilation the spacing between kernel elements: Shape(depth, height, width)
     * @return the output of the conv3d operation
     */
    public static NDList conv3d(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation) {
        return conv3d(input, weight, bias, stride, padding, dilation, 1);
    }

    /**
     * Applies 3D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, depth, height,
     *     width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, depth,
     *     height, width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(depth, height, width)
     * @param padding implicit paddings on both sides of the input: Shape(depth, height, width)
     * @param dilation the spacing between kernel elements: Shape(depth, height, width)
     * @param groups split input into groups: input channel(input.size(1)) should be divisible by
     *     the number of groups
     * @return the output of the conv3d operation
     */
    public static NDList conv3d(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 5 && weight.getShape().dimension() == 5,
                "the shape of input or weight doesn't match the conv2d");
        Preconditions.checkArgument(
                stride.dimension() == 3 && padding.dimension() == 3 && dilation.dimension() == 3,
                "the shape of stride or padding or dilation doesn't match the conv2d");
        return Convolution.convolution(input, weight, bias, stride, padding, dilation, groups);
    }

    /**
     * Creates a builder to build a {@code Conv3d}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Conv3d} type of {@link Block}. */
    public static final class Builder extends ConvolutionBuilder<Builder> {

        /** Creates a builder that can build a {@link Conv3d} block. */
        Builder() {
            stride = new Shape(1, 1, 1);
            padding = new Shape(0, 0, 0);
            dilation = new Shape(1, 1, 1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Conv3d} block.
         *
         * @return the {@link Conv3d} block
         */
        public Conv3d build() {
            validate();
            return new Conv3d(this);
        }
    }
}
