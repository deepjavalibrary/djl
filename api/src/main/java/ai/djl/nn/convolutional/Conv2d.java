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
 * Being the pioneer of convolution layers, {@code Conv2d} layer works on two dimensions of input,
 * {@link LayoutType#WIDTH} and {@link LayoutType#HEIGHT} as usually a {@code Conv2d} layer is used
 * to process data with two spatial dimensions, namely image. The concept itself works just as how
 * {@link Convolution} does, and each filter slides through an input data by two directions, first
 * traversing the {@link LayoutType#WIDTH} then traverses each row of the data.
 *
 * <p>First proposed by LeCun <i>et al.</i>'s <a
 * href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf">paper</a>, 2-dimensional convolution
 * layer gained its rising interest with the publication of <a
 * href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">
 * paper</a> about AlexNet for image classification task. It is still commonly used in image-related
 * tasks and adapted in other tasks, including but not limited to 1-dimensional data which may be
 * transformed to 2-dimensional data, though {@link Conv1d} is now available for use.
 *
 * <p>The input to a {@code Conv2d} is an {@link ai.djl.ndarray.NDList} with a single 4-D {@link
 * ai.djl.ndarray.NDArray}. The layout of the {@link ai.djl.ndarray.NDArray} must be "NCHW". The
 * shapes are
 *
 * <ul>
 *   <li>{@code data: (batch_size, channel, height, width)}
 *   <li>{@code weight: (num_filter, channel, kernel[0], kernel[1])}
 *   <li>{@code bias: (num_filter,)}
 *   <li>{@code out: (batch_size, num_filter, out_height, out_width)} <br>
 *       {@code out_height = f(height, kernel[0], pad[0], stride[0], dilate[0])} <br>
 *       {@code out_width = f(width, kernel[1], pad[1], stride[1], dilate[1])} <br>
 *       {@code where f(x, k, p, s, d) = floor((x + 2 * p - d * (k - 1) - 1)/s) + 1}
 * </ul>
 *
 * <p>Both {@code weight} and {@code bias} are learn-able parameters.
 *
 * @see Convolution
 */
public class Conv2d extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.HEIGHT, LayoutType.WIDTH
    };

    private static final String STRING_LAYOUT = "NCHW";
    private static final int NUM_DIMENSIONS = 4;

    Conv2d(Builder builder) {
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
     * Applies 2D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @return the output of the conv2d operation
     */
    public static NDList conv2d(NDArray input, NDArray weight) {
        return conv2d(input, weight, null, new Shape(1, 1), new Shape(0, 0), new Shape(1, 1));
    }

    /**
     * Applies 2D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @return the output of the conv2d operation
     */
    public static NDList conv2d(NDArray input, NDArray weight, NDArray bias) {
        return conv2d(input, weight, bias, new Shape(1, 1), new Shape(0, 0), new Shape(1, 1));
    }

    /**
     * Applies 2D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(height, width)
     * @return the output of the conv2d operation
     */
    public static NDList conv2d(NDArray input, NDArray weight, NDArray bias, Shape stride) {
        return conv2d(input, weight, bias, stride, new Shape(0, 0), new Shape(1, 1));
    }

    /**
     * Applies 2D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(height, width)
     * @param padding implicit paddings on both sides of the input: Shape(height, width)
     * @return the output of the conv2d operation
     */
    public static NDList conv2d(
            NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding) {
        return conv2d(input, weight, bias, stride, padding, new Shape(1, 1));
    }

    /**
     * Applies 2D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(height, width)
     * @param padding implicit paddings on both sides of the input: Shape(height, width)
     * @param dilation the spacing between kernel elements: Shape(height, width)
     * @return the output of the conv2d operation
     */
    public static NDList conv2d(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation) {
        return conv2d(input, weight, bias, stride, padding, dilation, 1);
    }

    /**
     * Applies 2D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, height, width)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, height,
     *     width)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(height, width)
     * @param padding implicit paddings on both sides of the input: Shape(height, width)
     * @param dilation the spacing between kernel elements: Shape(height, width)
     * @param groups split input into groups: input channel(input.size(1)) should be divisible by
     *     the number of groups
     * @return the output of the conv2d operation
     */
    public static NDList conv2d(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 4 && weight.getShape().dimension() == 4,
                "the shape of input or weight doesn't match the conv2d");
        Preconditions.checkArgument(
                stride.dimension() == 2 && padding.dimension() == 2 && dilation.dimension() == 2,
                "the shape of stride or padding or dilation doesn't match the conv2d");
        return Convolution.convolution(input, weight, bias, stride, padding, dilation, groups);
    }

    /**
     * Creates a builder to build a {@code Conv2d}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Conv2d} type of {@link Block}. */
    public static final class Builder extends ConvolutionBuilder<Builder> {

        /** Creates a builder that can build a {@link Conv2d} block. */
        Builder() {
            stride = new Shape(1, 1);
            padding = new Shape(0, 0);
            dilation = new Shape(1, 1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Conv2d} block.
         *
         * @return the {@link Conv2d} block
         */
        public Conv2d build() {
            validate();
            return new Conv2d(this);
        }
    }
}
