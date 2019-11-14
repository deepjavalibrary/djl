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
 * Computes 2-D convolution on 4-D input.
 *
 * <p>The input to a {@code Conv2D} is an {@link ai.djl.ndarray.NDList} with a single 4-D {@link
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
 */
public class Conv2D extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.HEIGHT, LayoutType.WIDTH
    };

    private static final String STRING_LAYOUT = "NCHW";
    private static final int NUM_DIMENSIONS = 4;
    private static final byte VERSION = 1;

    Conv2D(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    protected byte getVersion() {
        return VERSION;
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

    /** The Builder to construct a {@link Conv2D} type of {@link Block}. */
    public static final class Builder extends ConvolutionBuilder<Builder> {
        /** Creates a builder that can build a {@link Conv2D} block. */
        public Builder() {
            stride = new Shape(1, 1);
            pad = new Shape(0, 0);
            dilate = new Shape(1, 1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds a {@link Conv2D} block.
         *
         * @return the {@link Conv2D} block
         */
        public Conv2D build() {
            validate();
            return new Conv2D(this);
        }
    }
}
