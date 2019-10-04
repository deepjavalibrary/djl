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
package software.amazon.ai.nn.convolutional;

import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.BlockList;
import software.amazon.ai.util.PairList;

public class Conv1D extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.WIDTH
    };

    private static final String STRING_LAYOUT = "NCW";
    private static final int NUM_DIMENSIONS = 3;
    private static final byte VERSION = 1;

    Conv1D(Builder builder) {
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

    @Override
    public PairList<String, Block> getChildren() {
        return new BlockList();
    }

    /** The Builder to construct a {@link Conv1D} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        public Builder() {
            stride = new Shape(1);
            pad = new Shape(0);
            dilate = new Shape(1);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        public Conv1D build() {
            validate();
            return new Conv1D(this);
        }
    }
}
