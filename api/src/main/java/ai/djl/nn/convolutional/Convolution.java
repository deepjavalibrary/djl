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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public abstract class Convolution extends ParameterBlock {

    protected Shape kernel;
    protected Shape stride;
    protected Shape pad;
    protected Shape dilate;
    protected int numFilters;
    protected int numGroups;
    protected boolean includeBias;

    protected Parameter weight;
    protected Parameter bias;

    public Convolution(BaseBuilder<?> builder) {
        kernel = builder.getKernel();
        stride = builder.getStride();
        pad = builder.getPad();
        dilate = builder.getDilate();
        numFilters = builder.getNumFilters();
        numGroups = builder.getNumGroups();
        includeBias = builder.isIncludeBias();

        weight = new Parameter("weight", this, ParameterType.WEIGHT);
        if (includeBias) {
            bias = new Parameter("bias", this, ParameterType.BIAS);
        }
    }

    protected abstract byte getVersion();

    protected abstract LayoutType[] getExpectedLayout();

    protected abstract String getStringLayout();

    protected abstract int numDimensions();

    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(parameterStore, inputs);
        NDArrayEx ex = inputs.get(0).getNDArrayInternal();
        return ex.convolution(
                inputs,
                kernel,
                stride,
                pad,
                numFilters,
                numGroups,
                getStringLayout(),
                !includeBias,
                params);
    }

    @Override
    protected void beforeInitialize(Shape[] inputs) {
        Shape inputShape = inputs[0];
        Block.validateLayout(getExpectedLayout(), inputShape.getLayout());
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        long[] shape = new long[numDimensions()];
        shape[0] = inputs[0].get(0);
        shape[1] = numFilters;
        for (int i = 0; i < numDimensions() - 2; i++) {
            shape[2 + i] =
                    (inputs[0].get(2 + i)
                                            + 2 * pad.get(i)
                                            - dilate.get(0) * (kernel.get(i) - 1)
                                            - 1)
                                    / stride.get(0)
                            + 1;
        }
        return new Shape[] {new Shape(shape)};
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        Shape shape = inputShapes[0];
        switch (name) {
            case "weight":
                return new Shape(numFilters, shape.get(1)).addAll(kernel);
            case "bias":
                return new Shape(numFilters);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    @Override
    public List<Parameter> getDirectParameters() {
        List<Parameter> parameters = new ArrayList<>();
        parameters.add(weight);
        if (includeBias) {
            parameters.add(bias);
        }
        return parameters;
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(getVersion());
        weight.save(os);
        if (bias != null) {
            bias.save(os);
        }
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != getVersion()) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        weight.load(manager, is);
        if (bias != null) {
            bias.load(manager, is);
        }
    }

    private NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        NDArray data = inputs.head();
        Device device = data.getDevice();
        NDList ret = new NDList(3);
        ret.add(data);
        ret.add(parameterStore.getValue(weight, device));
        if (bias != null) {
            ret.add(parameterStore.getValue(bias, device));
        }
        return ret;
    }

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected Shape kernel;
        protected Shape stride;
        protected Shape pad;
        protected Shape dilate;
        protected int numFilters;
        protected int numGroups = 1;
        protected boolean includeBias = true;

        public Shape getKernel() {
            return kernel;
        }

        public Shape getStride() {
            return stride;
        }

        public Shape getPad() {
            return pad;
        }

        public Shape getDilate() {
            return dilate;
        }

        public int getNumFilters() {
            return numFilters;
        }

        public int getNumGroups() {
            return numGroups;
        }

        public boolean isIncludeBias() {
            return includeBias;
        }

        /**
         * Sets the shape of the kernel.
         *
         * @param kernel Shape of the kernel
         * @return Returns this Builder
         */
        public T setKernel(Shape kernel) {
            this.kernel = kernel;
            return self();
        }

        /**
         * Sets the stride of the convolution. Defaults to 1 in each dimension.
         *
         * @param stride Shape of the stride
         * @return Returns this Builder
         */
        public T setStride(Shape stride) {
            this.stride = stride;
            return self();
        }

        /**
         * Sets the padding along each dimension. Defaults to zero along each dimension
         *
         * @param pad Padding along each dimension
         * @return Returns this Builder
         */
        public T setPad(Shape pad) {
            this.pad = pad;
            return self();
        }

        /**
         * Sets the padding along each dimension. Defaults to zero along each dimension
         *
         * @param dilate Padding along each dimension
         * @return Returns this Builder
         */
        public T setDilate(Shape dilate) {
            this.dilate = dilate;
            return self();
        }

        /**
         * Sets the <b>Required</b> number of filters.
         *
         * @param numFilters Number of convolution filter(channel)
         * @return Returns this Builder
         */
        public T setNumFilters(int numFilters) {
            this.numFilters = numFilters;
            return self();
        }

        /**
         * Sets the number of group partitions.
         *
         * @param numGroups Number of group partitions
         * @return Returns this Builder
         */
        public T setNumGroups(int numGroups) {
            this.numGroups = numGroups;
            return self();
        }

        /**
         * Sets the optional parameter of whether to include a bias vector. Includes bias by
         * default.
         *
         * @param includeBias Whether to use a bias vector parameter
         * @return Returns this Builder
         */
        public T setBias(boolean includeBias) {
            this.includeBias = includeBias;
            return self();
        }

        protected void validate() {
            if (kernel == null || numFilters == 0) {
                throw new IllegalArgumentException("Kernel and numFilters must be set");
            }
        }

        protected abstract T self();
    }
}
