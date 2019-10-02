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
package software.amazon.ai.nn.convolutional;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.AbstractBlock;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.util.PairList;

public abstract class Convolution extends AbstractBlock {

    protected Shape kernel;
    protected Shape stride;
    protected Shape pad;
    protected Shape dilate;
    protected int numFilters;
    protected int numGroups;
    protected String layout;
    protected boolean includeBias;

    protected Parameter weight;
    protected Parameter bias;

    protected abstract byte getVersion();

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(inputs);
        NDArrayEx ex = inputs.get(0).getNDArrayInternal();
        return ex.convolution(
                inputs, kernel, stride, pad, numFilters, numGroups, layout, !includeBias, params);
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

    private NDList opInputs(NDList inputs) {
        initialize(inputs);
        NDArray data = inputs.head();
        if (bias == null) {
            return new NDList(data, weight.getArray());
        }
        return new NDList(data, weight.getArray(), bias.getArray());
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

        protected abstract T self();
    }
}
