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

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;

/**
 * Transposed convolution, also named fractionally-strided convolution <a
 * href="https://arxiv.org/pdf/1603.07285">Dumoulin &amp; Visin</a> or deconvolution <a
 * href="https://ieeexplore.ieee.org/document/7298965">Long et al., 2015</a>, serves this purpose.
 *
 * <p>The need for transposed convolutions generally arises from the desire to use a transformation
 * going in the opposite direction of a normal convolution, i.e., from something that has the shape
 * of the output of some convolution to something that has the shape of its input while maintaining
 * a connectivity pattern that is compatible with said convolution.
 *
 * <p>Current implementations of {@code Deconvolution} are {@link Conv1dTranspose} with input
 * dimension of {@link LayoutType#WIDTH} and {@link Conv2dTranspose} with input dimension of {@link
 * LayoutType#WIDTH} and {@link LayoutType#HEIGHT}. These implementations share the same core
 * principal as a {@code Deconvolution} layer does, with the difference being the number of input
 * dimension each operates on as denoted by {@code ConvXdTranspose} for {@code X} dimension(s).
 */
public abstract class Deconvolution extends AbstractBlock {

    protected Shape kernelShape;
    protected Shape stride;
    protected Shape padding;
    protected Shape outPadding;
    protected Shape dilation;
    protected int filters;
    protected int groups;
    protected boolean includeBias;

    protected Parameter weight;
    protected Parameter bias;

    /**
     * Creates a {@link Deconvolution} object.
     *
     * @param builder the {@code Builder} that has the necessary configurations
     */
    public Deconvolution(DeconvolutionBuilder<?> builder) {
        kernelShape = builder.kernelShape;
        stride = builder.stride;
        padding = builder.padding;
        outPadding = builder.outPadding;
        dilation = builder.dilation;
        filters = builder.filters;
        groups = builder.groups;
        includeBias = builder.includeBias;

        weight =
                addParameter(
                        Parameter.builder()
                                .setName("weight")
                                .setType(Parameter.Type.WEIGHT)
                                .build());
        if (includeBias) {
            bias =
                    addParameter(
                            Parameter.builder()
                                    .setName("bias")
                                    .setType(Parameter.Type.BIAS)
                                    .build());
        }
    }

    /**
     * Returns the expected layout of the input.
     *
     * @return the expected layout of the input
     */
    protected abstract LayoutType[] getExpectedLayout();

    /**
     * Returns the string representing the layout of the input.
     *
     * @return the string representing the layout of the input
     */
    protected abstract String getStringLayout();

    /**
     * Returns the number of dimensions of the input.
     *
     * @return the number of dimensions of the input
     */
    protected abstract int numDimensions();

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray weightArr = parameterStore.getValue(weight, device, training);
        NDArray biasArr = parameterStore.getValue(bias, device, training);
        return deconvolution(
                input, weightArr, biasArr, stride, padding, outPadding, dilation, groups);
    }

    /** {@inheritDoc} */
    @Override
    protected void beforeInitialize(Shape... inputShapes) {
        super.beforeInitialize(inputShapes);
        Block.validateLayout(getExpectedLayout(), inputShapes[0].getLayout());
    }

    /** {@inheritDoc} */
    @Override
    protected void prepare(Shape[] inputs) {
        long inputChannel = inputs[0].get(1);
        weight.setShape(new Shape(filters, inputChannel / groups).addAll(kernelShape));
        if (bias != null) {
            bias.setShape(new Shape(filters));
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        long[] shape = new long[numDimensions()];
        shape[0] = inputs[0].get(0);
        shape[1] = filters;
        for (int i = 0; i < numDimensions() - 2; i++) {
            shape[2 + i] =
                    (inputs[0].get(2 + i) - 1) * stride.get(i)
                            - 2 * padding.get(i)
                            + dilation.get(i) * (kernelShape.get(i) - 1)
                            + outPadding.get(i)
                            + 1;
        }
        return new Shape[] {new Shape(shape)};
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == version) {
            readInputShapes(is);
        } else {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }

    /**
     * Applies N-D deconvolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, ...)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, ...)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the deconvolving kernel: Shape(w) or Shape(h, w)
     * @param padding implicit paddings on both sides of the input: Shape(w) or Shape(h, w)
     * @param outPadding Controls the amount of implicit zero-paddings on both sides of the output
     *     for output_padding number of points for each dimension. Shape(w) or Shape(h, w)
     * @param dilation the spacing between kernel elements: Shape(w) or Shape(h, w)
     * @param groups split input into groups: input channel(input.size(1)) should be divisible by
     *     the number of groups
     * @return the output of the deconvolution operation
     */
    static NDList deconvolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups) {
        return input.getNDArrayInternal()
                .deconvolution(input, weight, bias, stride, padding, outPadding, dilation, groups);
    }
    /**
     * A builder that can build any {@code Deconvolution} block.
     *
     * @param <T> the type of {@code Deconvolution} block to build
     */
    @SuppressWarnings("rawtypes")
    public abstract static class DeconvolutionBuilder<T extends DeconvolutionBuilder> {

        protected Shape kernelShape;
        protected Shape stride;
        protected Shape padding;
        protected Shape outPadding;
        protected Shape dilation;
        protected int filters;
        protected int groups = 1;
        protected boolean includeBias = true;

        /**
         * Sets the shape of the kernel.
         *
         * @param kernelShape the shape of the kernel
         * @return this Builder
         */
        public T setKernelShape(Shape kernelShape) {
            this.kernelShape = kernelShape;
            return self();
        }

        /**
         * Sets the stride of the deconvolution. Defaults to 1 in each dimension.
         *
         * @param stride the shape of the stride
         * @return this Builder
         */
        public T optStride(Shape stride) {
            this.stride = stride;
            return self();
        }

        /**
         * Sets the padding along each dimension. Defaults to 0 along each dimension.
         *
         * @param padding the shape of padding along each dimension
         * @return this Builder
         */
        public T optPadding(Shape padding) {
            this.padding = padding;
            return self();
        }
        /**
         * Sets the out_padding along each dimension. Defaults to 0 along each dimension.
         *
         * @param outPadding the shape of out_padding along each dimension
         * @return this Builder
         */
        public T optOutPadding(Shape outPadding) {
            this.outPadding = outPadding;
            return self();
        }
        /**
         * Sets the dilation along each dimension. Defaults to 1 along each dimension.
         *
         * @param dilate the shape of dilation along each dimension
         * @return this Builder
         */
        public T optDilation(Shape dilate) {
            this.dilation = dilate;
            return self();
        }

        /**
         * Sets the <b>Required</b> number of filters.
         *
         * @param filters the number of deconvolution filters(channels)
         * @return this Builder
         */
        public T setFilters(int filters) {
            this.filters = filters;
            return self();
        }

        /**
         * Sets the number of group partitions.
         *
         * @param groups the number of group partitions
         * @return this Builder
         */
        public T optGroups(int groups) {
            this.groups = groups;
            return self();
        }

        /**
         * Sets the optional parameter of whether to include a bias vector. Includes bias by
         * default.
         *
         * @param includeBias whether to use a bias vector parameter
         * @return this Builder
         */
        public T optBias(boolean includeBias) {
            this.includeBias = includeBias;
            return self();
        }

        /**
         * Validates that the required arguments are set.
         *
         * @throws IllegalArgumentException if the required arguments are not set
         */
        protected void validate() {
            if (kernelShape == null || filters == 0) {
                throw new IllegalArgumentException("Kernel and numFilters must be set");
            }
        }

        protected abstract T self();
    }
}
