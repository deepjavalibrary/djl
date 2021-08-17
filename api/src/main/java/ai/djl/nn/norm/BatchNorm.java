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
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
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
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-modern/batch-norm.html">The D2L chapter on
 *     batch normalization</a>
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

        // make gamma trainable if scale
        gamma =
                addParameter(
                        Parameter.builder()
                                .setName("gamma")
                                .setType(Parameter.Type.GAMMA)
                                .optRequiresGrad(scale)
                                .build());
        // make beta trainable if center
        beta =
                addParameter(
                        Parameter.builder()
                                .setName("beta")
                                .setType(Parameter.Type.BETA)
                                .optRequiresGrad(center)
                                .build());
        runningMean =
                addParameter(
                        Parameter.builder()
                                .setName("runningMean")
                                .setType(Parameter.Type.RUNNING_MEAN)
                                .optRequiresGrad(false)
                                .build());
        runningVar =
                addParameter(
                        Parameter.builder()
                                .setName("runningVar")
                                .setType(Parameter.Type.RUNNING_VAR)
                                .optRequiresGrad(false)
                                .build());
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray gammaArr = parameterStore.getValue(gamma, device, training);
        NDArray betaArr = parameterStore.getValue(beta, device, training);
        NDArray runningMeanArr = parameterStore.getValue(runningMean, device, training);
        NDArray runningVarArr = parameterStore.getValue(runningVar, device, training);
        return batchNorm(
                input,
                runningMeanArr,
                runningVarArr,
                gammaArr,
                betaArr,
                axis,
                momentum,
                epsilon,
                training);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {inputShapes[0]};
    }

    /** {@inheritDoc} */
    @Override
    protected void beforeInitialize(Shape... inputShapes) {
        super.beforeInitialize(inputShapes);
        inChannels = inputShapes[0].size(axis);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Shape[] inputShapes) {
        gamma.setShape(new Shape(inChannels));
        beta.setShape(new Shape(inChannels));
        runningMean.setShape(new Shape(inChannels));
        runningVar.setShape(new Shape(inChannels));
    }

    /** {@inheritDoc} */
    @Override
    protected void saveMetadata(DataOutputStream os) throws IOException {
        saveInputShapes(os);
        os.writeLong(inChannels);
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == VERSION) {
            readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        inChannels = is.readLong();
    }

    /**
     * Applies Batch Normalization for each channel across a batch of data.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, *), * could be
     *     empty, width, (height, width), (depth, height, width)
     * @param runningMean runningMean {@code NDArray}
     * @param runningVar runningVar {@code NDArray}
     * @return the output {@code NDArray} of shape (batchSize, inputChannel, *), * could be empty,
     *     width, (height, width), (depth, height, width)
     */
    public static NDList batchNorm(NDArray input, NDArray runningMean, NDArray runningVar) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(input, runningMean, runningVar, null, null, 1, 0.9f, 1E-5f, true);
    }

    /**
     * Applies Batch Normalization for each channel across a batch of data.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, *), * could be
     *     empty, width, (height, width), (depth, height, width)
     * @param runningMean runningMean {@code NDArray}
     * @param runningVar runningVar {@code NDArray}
     * @param gamma gamma weight {@code NDArray}
     * @param beta beta weight {@code NDArray}
     * @return the output {@code NDArray} of shape (batchSize, inputChannel, *), * could be empty,
     *     width, (height, width), (depth, height, width)
     */
    public static NDList batchNorm(
            NDArray input, NDArray runningMean, NDArray runningVar, NDArray gamma, NDArray beta) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(input, runningMean, runningVar, gamma, beta, 1, 0.9f, 1E-5f, true);
    }

    /**
     * Applies Batch Normalization for each channel across a batch of data.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, *), * could be
     *     empty, width, (height, width), (depth, height, width)
     * @param runningMean runningMean {@code NDArray}
     * @param runningVar runningVar {@code NDArray}
     * @param gamma gamma weight {@code NDArray}
     * @param beta beta weight {@code NDArray}
     * @param axis the axis that should be normalized
     * @return the output {@code NDArray} of shape (batchSize, inputChannel, *), * could be empty,
     *     width, (height, width), (depth, height, width)
     */
    public static NDList batchNorm(
            NDArray input,
            NDArray runningMean,
            NDArray runningVar,
            NDArray gamma,
            NDArray beta,
            int axis) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(input, runningMean, runningVar, gamma, beta, axis, 0.9f, 1E-5f, true);
    }

    /**
     * Applies Batch Normalization for each channel across a batch of data.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, *), * could be
     *     empty, width, (height, width), (depth, height, width)
     * @param runningMean runningMean {@code NDArray}
     * @param runningVar runningVar {@code NDArray}
     * @param gamma gamma weight {@code NDArray}
     * @param beta beta weight {@code NDArray}
     * @param axis the axis that should be normalized
     * @param momentum the value used for the runningMean and runningVar computation.
     * @param eps a value added to the denominator for numerical stability
     * @param training indicate the training mode if true
     * @return the output {@code NDArray} of shape (batchSize, inputChannel, *), * could be empty,
     *     width, (height, width), (depth, height, width)
     */
    public static NDList batchNorm(
            NDArray input,
            NDArray runningMean,
            NDArray runningVar,
            NDArray gamma,
            NDArray beta,
            int axis,
            float momentum,
            float eps,
            boolean training) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(
                input, runningMean, runningVar, gamma, beta, axis, momentum, eps, training);
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
         * @param axis the axis in which channel is specified
         * @return this Builder
         */
        public Builder optAxis(int axis) {
            this.axis = axis;
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
