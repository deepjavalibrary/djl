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

import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;

public class Conv1D extends Convolution {

    private static final LayoutType[] EXPECTED_LAYOUT = {
        LayoutType.BATCH, LayoutType.CHANNEL, LayoutType.WIDTH
    };

    private static final String LAYOUT = "NCW";
    private static final byte VERSION = 1;

    Conv1D(Builder builder) {
        kernel = builder.getKernel();
        stride = builder.getStride() == null ? new Shape(1) : builder.getStride();
        pad = builder.getPad() == null ? new Shape(0) : builder.getPad();
        dilate = builder.getDilate() == null ? new Shape(1) : builder.getDilate();
        numFilters = builder.getNumFilters();
        numGroups = builder.getNumGroups();
        layout = LAYOUT;
        includeBias = builder.isIncludeBias();

        weight = new Parameter("weight", this, ParameterType.WEIGHT);
        if (includeBias) {
            bias = new Parameter("bias", this, ParameterType.BIAS);
        }
    }

    @Override
    protected byte getVersion() {
        return VERSION;
    }

    @Override
    protected void beforeInitialize(Shape[] inputs) {
        Shape inputShape = inputs[0];
        Block.validateLayout(EXPECTED_LAYOUT, inputShape.getLayout());
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        long batchSize = inputs[0].get(0);
        long outWidth =
                (inputs[0].get(2) + 2 * pad.get(0) - dilate.get(0) * (kernel.get(0) - 1) - 1)
                                / stride.get(0)
                        + 1;
        return new Shape[] {new Shape(batchSize, numFilters, outWidth)};
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        switch (name) {
            case "weight":
                return new Shape(numFilters, inputShape.get(1), kernel.get(0));
            case "bias":
                return new Shape(numFilters);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    /** The Builder to construct a {@link Conv1D} type of {@link Block}. */
    public static final class Builder extends BaseBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        public Conv1D build() {
            if (kernel == null || numFilters == 0) {
                throw new IllegalArgumentException("Kernel and numFilters must be set");
            }
            return new Conv1D(this);
        }
    }
}
