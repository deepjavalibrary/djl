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
package ai.djl.nn.norm;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class BatchNorm extends ParameterBlock {

    private static final byte VERSION = 1;

    private int axis;
    private float epsilon;
    private float momentum;
    private long inChannels;

    private Parameter runningMean;
    private Parameter runningVar;

    BatchNorm(Builder builder) {
        axis = builder.getAxis();
        epsilon = builder.getEpsilon();
        momentum = builder.getMomentum();
        runningMean = new Parameter("runningMean", this, ParameterType.RUNNING_MEAN, false);
        runningVar = new Parameter("runningVar", this, ParameterType.RUNNING_VAR, false);
    }

    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(parameterStore, inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        return ex.batchNorm(inputs, epsilon, momentum, axis, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[] {inputShapes[0]};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Arrays.asList(runningMean, runningVar);
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(Shape[] inputShapes) {
        this.inputShapes = inputShapes;
        inChannels = inputShapes[0].size(axis);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        switch (name) {
            case "runningMean":
            case "runningVar":
                return new Shape(inChannels);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    private NDList opInputs(ParameterStore parameterStore, NDList inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Linear requires exactly 1 NDArray");
        }
        NDArray data = inputs.singletonOrThrow();
        Device device = data.getDevice();
        NDArray gamma = data.getManager().ones(new Shape(inChannels));
        NDArray beta = data.getManager().zeros(new Shape(inChannels));
        NDArray runningMeanValue = parameterStore.getValue(runningMean, device);
        NDArray runningVarValue = parameterStore.getValue(runningVar, device);
        return new NDList(data, gamma, beta, runningMeanValue, runningVarValue);
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        os.writeLong(inChannels);
        runningMean.save(os);
        runningVar.save(os);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        inChannels = is.readLong();
        runningMean.load(manager, is);
        runningVar.load(manager, is);
    }

    public static final class Builder {

        private int axis = 1;
        private float epsilon = 1E-5f;
        private float momentum = .9f;

        public int getAxis() {
            return axis;
        }

        public float getEpsilon() {
            return epsilon;
        }

        public float getMomentum() {
            return momentum;
        }

        public Builder optAxis(int val) {
            axis = val;
            return this;
        }

        public Builder optEpsilon(float val) {
            epsilon = val;
            return this;
        }

        public Builder optMomentum(float val) {
            momentum = val;
            return this;
        }

        public BatchNorm build() {
            return new BatchNorm(this);
        }
    }
}
