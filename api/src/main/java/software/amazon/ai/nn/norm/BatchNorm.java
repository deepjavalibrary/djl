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
package software.amazon.ai.nn.norm;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.AbstractBlock;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.PairList;

public class BatchNorm extends AbstractBlock {

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
        runningMean = new Parameter("runningMean", this, ParameterType.OTHER, Initializer.ONES);
        runningVar = new Parameter("runningVar", this, ParameterType.OTHER, Initializer.ONES);
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        inputs = opInputs(inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        return ex.batchNorm(inputs, epsilon, momentum, axis, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getOutputShape(Shape... inputs) {
        return inputs[0];
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Arrays.asList(runningMean, runningVar);
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(NDList inputs) {
        inChannels = inputs.get(0).size(axis);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        switch (name) {
            case "runningMean":
            case "runningVar":
                return new Shape(inChannels);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    private NDList opInputs(NDList inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Linear requires exactly 1 NDArray");
        }
        ensureInitialized(inputs);
        NDArray data = inputs.get(0);
        NDArray gamma = data.getManager().ones(new Shape(inChannels));
        NDArray beta = data.getManager().zeros(new Shape(inChannels));
        return new NDList(data, gamma, beta, runningMean.getArray(), runningVar.getArray());
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        runningMean.save(os);
        runningVar.save(os);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
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

        public Builder setAxis(int val) {
            axis = val;
            return this;
        }

        public Builder setEpsilon(float val) {
            epsilon = val;
            return this;
        }

        public Builder setMomentum(float val) {
            momentum = val;
            return this;
        }

        public BatchNorm build() {
            return new BatchNorm(this);
        }
    }
}
