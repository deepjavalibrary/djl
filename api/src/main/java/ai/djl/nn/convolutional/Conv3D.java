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

public class Conv3D extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.DEPTH, LayoutType.HEIGHT, LayoutType.WIDTH
    };
    private static final String STRING_LAYOUT = "NCDHW";
    private static final int NUM_DIMENSIONS = 5;
    private static final byte VERSION = 1;

    Conv3D(Builder builder) {
        super(builder);
    }

    @Override
    protected byte getVersion() {
        return VERSION;
    }

    @Override
    protected LayoutType[] getExpectedLayout() {
        return EXPECTED_LAYOUT;
    }

    @Override
    protected String getStringLayout() {
        return STRING_LAYOUT;
    }

    @Override
    protected int numDimensions() {
        return NUM_DIMENSIONS;
    }

    /** The Builder to construct a {@link Conv3D} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        public Builder() {
            stride = new Shape(1, 1, 1);
            pad = new Shape(0, 0, 0);
            dilate = new Shape(1, 1, 1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        public Conv3D build() {
            validate();
            return new Conv3D(this);
        }
    }
}
