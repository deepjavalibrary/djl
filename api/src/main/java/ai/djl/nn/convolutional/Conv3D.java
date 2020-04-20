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

import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

/**
 * <p>{@code Conv3D} layer behaves just as {@link Convolution} does, with the distinction being it
 * operates of 3-dimensional data such as medical images or video data. The traversal of
 * each filter begins from {@link LayoutType#WIDTH} then to {@link LayoutType#HEIGHT}, and lastly
 * across each {@link LayoutType#DEPTH} in the specified {@code depth} size of the data.
 *
 * <p>The utilization of {@code Conv3D} layer allows deeper analysis of visual data such as those
 * in medical images, or even analysis on temporal data such as video data as a whole instead of
 * processing each frame with a {@link Conv2D} layer, despite this being a common practice in
 * computer vision researches. The benefit of utilizing this kind of layer is the maintaining of
 * serial data across 2-dimensional data, hence could be beneficial for research focus on such as
 * object tracking. The drawback is that this kind of layer is more costly compared to other
 * convolution layer types since dot product operation is performed on all three dimensions.
 *
 * <p>The input to a {@code Conv3D} is an {@link ai.djl.ndarray.NDList} with a single 5-D {@link
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
 */
public class Conv3D extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.DEPTH, LayoutType.HEIGHT, LayoutType.WIDTH
    };
    private static final String STRING_LAYOUT = "NCDHW";
    private static final int NUM_DIMENSIONS = 5;

    Conv3D(Builder builder) {
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
     * Creates a builder to build a {@code Conv3D}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Conv3D} type of {@link Block}. */
    public static final class Builder extends ConvolutionBuilder<Builder> {

        /** Creates a builder that can build a {@link Conv3D} block. */
        Builder() {
            stride = new Shape(1, 1, 1);
            pad = new Shape(0, 0, 0);
            dilate = new Shape(1, 1, 1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Conv3D} block.
         *
         * @return the {@link Conv3D} block
         */
        public Conv3D build() {
            validate();
            return new Conv3D(this);
        }
    }
}
