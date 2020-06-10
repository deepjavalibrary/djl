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
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * In batch training (training with more than one samples per iteration), a batch normalization
 * layer works by normalizing the values of input data to have mean of 0 and variance of 1. Since
 * this may alter the representation of a layer, two parameters (\ (\gamma\) and \(\beta\)) are
 * learned along the normalization process to respectively scale and shift the normalized output
 * (activations) to have any mean and variance so the network can utilize non-linear transformations
 * such as sigmoid function as described in the <a
 * href="https://arxiv.org/abs/1502.03167">paper</a>. During backpropagation, both \(\gamma\) and
 * \(\beta\) parameters are included following the chain-rule in derivation.
 *
 * <p>The problem of varying distribution of input data requires the training process of a deep
 * network to compensate for each different data distribution per batch, hence changing parameters'
 * values as new batch data is processed and changes distribution of the network's (and each of its
 * layers) activations. This condition is termed as internal covariate shift, and such occurrence
 * prevents the network to learn faster and generalize better to unseen data.
 *
 * <p>With batch normalization, one benefits by having faster learning process as batch
 * normalization allows larger learning rate without causing gradient problems on backpropagation as
 * all inputs are normalized and hence reducing the scale of weight update impact on
 * backpropagation. In some cases, the utilization of batch normalization layer regularizes the
 * network and reduces, even eliminates, the need for dropout, which in turn results in even faster
 * training process since dropout slows down training by 2-3 times. However, it was reported that
 * batch normalization may not be beneficial when small batch sizes are used.
 *
 * <p>Formally, batch normalization is represented below: <br>
 * \(\hat{x} \:=\: \frac{x \:-\: \mu_{batch}}{\sqrt{\sigma^2_{batch} \:+\: \epsilon}}\), <br>
 * where \(\hat{x}\) is the normalized input, \(\mu_{batch}\) and \(\sigma^2_{batch}\) respectively
 * denote the mean and variance of a batch, and \(\epsilon\) (epsilon) is a constant for numerical
 * stability. The scale and shift operation can be formally defined as follows: <br>
 * \(y \:=\: \gamma\hat{x} \:+\: \beta\), <br>
 * where \(\gamma\) is the scale factor and \(\beta\) is the shift factor.
 */
public class BatchNorm extends AbstractBlock {

    private static final byte VERSION = 2;

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
        super(VERSION);
        axis = builder.axis;
        epsilon = builder.epsilon;
        momentum = builder.momentum;
        center = builder.center;
        scale = builder.scale;
        // When creating parameters we use a callback as "inChannels" is set before initialization,
        // it is not known yet.
        // make gamma trainable if scale
        gamma =
                addParameter(
                        new Parameter("gamma", this, ParameterType.GAMMA, scale),
                        (inputShapes) -> new Shape(inChannels));
        // make beta trainable if center
        beta =
                addParameter(
                        new Parameter("beta", this, ParameterType.BETA, center),
                        (inputShapes) -> new Shape(inChannels));
        runningMean =
                addParameter(
                        new Parameter("runningMean", this, ParameterType.RUNNING_MEAN, false),
                        (inputShapes) -> new Shape(inChannels));
        runningVar =
                addParameter(
                        new Parameter("runningVar", this, ParameterType.RUNNING_VAR, false),
                        (inputShapes) -> new Shape(inChannels));
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        inputs = opInputs(parameterStore, inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        return ex.batchNorm(inputs, epsilon, momentum, axis, center, scale, training, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[] {inputShapes[0]};
    }

    /** {@inheritDoc} */
    @Override
    public void beforeInitialize(Shape[] inputShapes) {
        this.inputShapes = inputShapes;
        inChannels = inputShapes[0].size(axis);
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
    protected void saveMetadata(DataOutputStream os) throws IOException {
        saveInputShapes(os);
        os.writeLong(inChannels);
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte version, DataInputStream is)
            throws IOException, MalformedModelException {
        if (version == VERSION) {
            readInputShapes(is);
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        inChannels = is.readLong();
    }

    /**
     * Creates a builder to build a {@code BatchNorm}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link BatchNorm}. */
    public static final class Builder {

        private int axis = 1;
        private float epsilon = 1E-5f;
        private float momentum = .9f;
        private boolean center = true;
        private boolean scale = true;

        Builder() {}

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
