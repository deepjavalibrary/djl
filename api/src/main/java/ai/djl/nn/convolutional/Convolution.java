/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
 * A convolution layer does a dot product calculation on each channel of \(k\)-channel input data by
 * specified number of filters, each containing \(k\) kernels for calculating each channel in the
 * input data and then summed per filter, hence the number of filters denote the number of output
 * channels of a convolution layer. Some modifications may be set on a convolution layer, namely
 * stride which shows the distance between each convolved input data in a channel, and padding which
 * shows the preservation of input size (width and/or height and/or depth) by adding specified
 * padding to the sides of the output. A convolution layer extracts features of input data with
 * different representations where each representation lies per channel in the output, often known
 * as feature map or feature vector.
 *
 * <p>While convolution process itself has been around for quite some time in mathematics, in 1998
 * LeCun <i>et al.</i> implemented the very first convolution layers forming a network called
 * LeNet-5 for character recognition task; details of the network's implementation can be find in
 * LeNet-5's <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf">paper</a>. When other
 * approaches at that time used handcrafted features with external stage of feature extraction,
 * convolution layer performed feature extraction on its own with no human interference. This marks
 * a new era of machine-extracted features, but it was not until 2012 that the published <a
 * href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">
 * paper</a> of AlexNet marked the beginning of convolutional neural networks, which by the name
 * itself heavily relies on convolution layer.
 *
 * <p>Convolution layer is usually used in image-related tasks due to its well-renowned performance
 * as shown by existing works and currently, other non-image-related fields of study are beginning
 * to incorporate convolution layer as an addition or replacement of previous approaches, with one
 * example being time series processing with 1-dimensional convolution layer. Due to the nature of
 * convolution that processes all points in the input data, it is computationally expensive, hence
 * the use of GPU is strongly recommended for faster performance as opposed to using CPU. Note that
 * it is also common to stack convolution layers with different output channels for more
 * representations of the input data.
 *
 * <p>Current implementations of {@code Convolution} are {@link Conv1d} with input dimension of
 * {@link LayoutType#WIDTH}, {@link Conv2d} with input dimension of {@link LayoutType#WIDTH} and
 * {@link LayoutType#HEIGHT}, and lastly {@link Conv3d} with input dimension of {@link
 * LayoutType#WIDTH}, {@link LayoutType#HEIGHT}, and {@link LayoutType#DEPTH}. These implementations
 * share the same core principal as a {@code Convolution} layer does, with the difference being the
 * number of input dimension each operates on as denoted by {@code ConvXD} for {@code X}
 * dimension(s).
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-neural-networks/why-conv.html">The D2L
 *     chapters on convolution</a>
 */
public abstract class Convolution extends AbstractBlock {

    private static final byte VERSION = 3;

    protected Shape kernelShape;
    protected Shape stride;
    protected Shape padding;
    protected Shape dilation;
    protected int filters;
    protected int groups;
    protected boolean includeBias;

    protected Parameter weight;
    protected Parameter bias;

    /**
     * Creates a {@link Convolution} object.
     *
     * @param builder the {@code Builder} that has the necessary configurations
     */
    public Convolution(ConvolutionBuilder<?> builder) {
        super(VERSION);
        kernelShape = builder.kernelShape;
        stride = builder.stride;
        padding = builder.padding;
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
        return convolution(input, weightArr, biasArr, stride, padding, dilation, groups);
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
                    (inputs[0].get(2 + i)
                                            + 2 * padding.get(i)
                                            - dilation.get(i) * (kernelShape.get(i) - 1)
                                            - 1)
                                    / stride.get(i)
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
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }

    /**
     * Applies N-D convolution over an input signal composed of several input planes.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, ...)
     * @param weight filters {@code NDArray} of shape (outChannel, inputChannel/groups, ...)
     * @param bias bias {@code NDArray} of shape (outChannel)
     * @param stride the stride of the convolving kernel: Shape(w), Shape(h, w) or Shape(d, h, w)
     * @param padding implicit paddings on both sides of the input: Shape(w), Shape(h, w) or
     *     Shape(d, h, w)
     * @param dilation the spacing between kernel elements: Shape(w), Shape(h, w) or Shape(d, h, w)
     * @param groups split input into groups: input channel(input.size(1)) should be divisible by
     *     the number of groups
     * @return the output of the convolution operation
     */
    static NDList convolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        return input.getNDArrayInternal()
                .convolution(input, weight, bias, stride, padding, dilation, groups);
    }
    /**
     * A builder that can build any {@code Convolution} block.
     *
     * @param <T> the type of {@code Convolution} block to build
     */
    @SuppressWarnings("rawtypes")
    public abstract static class ConvolutionBuilder<T extends ConvolutionBuilder> {

        protected Shape kernelShape;
        protected Shape stride;
        protected Shape padding;
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
         * Sets the stride of the convolution. Defaults to 1 in each dimension.
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
         * @param filters the number of convolution filters(channels)
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
