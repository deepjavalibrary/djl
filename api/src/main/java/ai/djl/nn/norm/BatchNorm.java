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
import ai.djl.MalformedModelException;
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

/**
 * Normalizes a data batch by mean and variance, and applies a scale gamma as well as offset beta.
 *
 * <p>See <a href="https://en.wikipedia.org/wiki/Batch_normalization">wikipedia</a> or the original
 * <a href="https://arxiv.org/abs/1502.03167">paper</a>.
 */
public class BatchNorm extends ParameterBlock {

    private static final byte VERSION = 1;

    private int axis;
    private float epsilon;
    private float momentum;
    private long inChannels;
    private boolean center;
    private boolean scale;

    private Parameter gamma;
    private Parameter beta;
    private Parameter runningMean;
    private Parameter runningVar;

    BatchNorm(Builder builder) {
        axis = builder.axis;
        epsilon = builder.epsilon;
        momentum = builder.momentum;
        center = builder.center;
        scale = builder.scale;
        // make gamma trainable if scale
        gamma = new Parameter("gamma", this, ParameterType.GAMMA, scale);
        // make beta trainable if center
        beta = new Parameter("beta", this, ParameterType.BETA, center);
        runningMean = new Parameter("runningMean", this, ParameterType.RUNNING_MEAN, false);
        runningVar = new Parameter("runningVar", this, ParameterType.RUNNING_VAR, false);
    }

    /** {@inheritDoc} */
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
        return Arrays.asList(gamma, beta, runningMean, runningVar);
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
            case "gamma":
            case "beta":
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
        NDArray gammaValue = parameterStore.getValue(gamma, device);
        NDArray betaValue = parameterStore.getValue(beta, device);
        NDArray runningMeanValue = parameterStore.getValue(runningMean, device);
        NDArray runningVarValue = parameterStore.getValue(runningVar, device);
        return new NDList(data, gammaValue, betaValue, runningMeanValue, runningVarValue);
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        os.writeLong(inChannels);
        gamma.save(os);
        beta.save(os);
        runningMean.save(os);
        runningVar.save(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        inChannels = is.readLong();
        gamma.load(manager, is);
        beta.load(manager, is);
        runningMean.load(manager, is);
        runningVar.load(manager, is);
    }

    /** The Builder to construct a {@link BatchNorm}. */
    public static final class Builder {

        private int axis = 1;
        private float epsilon = 1E-5f;
        private float momentum = .9f;
        private boolean center = true;
        private boolean scale = true;

        /**
         * Set the axis in which channel is specified. Defaults to 1.
         *
         * @param val the axis in which channel is specified
         * @return this Builder
         */
        public Builder optAxis(int val) {
            axis = val;
            return this;
        }

        /**
         * If True, add offset of `beta` to normalized tensor. Defaults to True.
         *
         * @param val True or False on whether to add and train offset value
         * @return this Builder
         */
        public Builder optCenter(boolean val) {
            center = val;
            return this;
        }

        /**
         * If True, multiply result by `gamma`. Defaults to True;
         *
         * @param val True or False on whether to add and train scale value
         * @return this Builder
         */
        public Builder optScale(boolean val) {
            scale = val;
            return this;
        }

        /**
         * Sets the epsilon value to prevent division by 0.
         *
         * @param val the epsilon value
         * @return this Builder
         */
        public Builder optEpsilon(float val) {
            epsilon = val;
            return this;
        }

        /**
         * Set the momentum for moving average.
         *
         * @param val the momentum for moving average
         * @return this Builder
         */
        public Builder optMomentum(float val) {
            momentum = val;
            return this;
        }

        /**
         * Builds a {@link BatchNorm} block.
         *
         * @return the {@link BatchNorm} block
         */
        public BatchNorm build() {
            return new BatchNorm(this);
        }
    }
}
