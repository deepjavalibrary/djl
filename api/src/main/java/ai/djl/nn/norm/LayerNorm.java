/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import java.util.Arrays;

/**
 * Layer normalization works by normalizing the values of input data for each input sample to have
 * mean of 0 and variance of 1. Since this may alter the representation of a layer, two parameters
 * (\ (\gamma\) and \(\beta\)) are learned along the normalization process to respectively scale and
 * shift the normalized output (activations) to have any mean and variance so the network can
 * utilize non-linear transformations such as sigmoid function as described in the <a
 * href="https://arxiv.org/abs/1607.06450">paper</a>. During backpropagation, both \(\gamma\) and
 * \(\beta\) parameters are included following the chain-rule in derivation.
 *
 * <p>Citing the abstract of the paper: "Training state-of-the-art, deep neural networks is
 * computationally expensive. One way to reduce the training time is to normalize the activities of
 * the neurons. A recently introduced technique called batch normalization uses the distribution of
 * the summed input to a neuron over a mini-batch of training cases to compute a mean and variance
 * which are then used to normalize the summed input to that neuron on each training case. This
 * significantly reduces the training time in feed-forward neural networks. However, the effect of
 * batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to
 * recurrent neural networks. In this paper, we transpose batch normalization into layer
 * normalization by computing the mean and variance used for normalization from all of the summed
 * inputs to the neurons in a layer on a single training case. Like batch normalization, we also
 * give each neuron its own adaptive bias and gain which are applied after the normalization but
 * before the non-linearity. Unlike batch normalization, layer normalization performs exactly the
 * same computation at training and test times. It is also straightforward to apply to recurrent
 * neural networks by computing the normalization statistics separately at each time step. Layer
 * normalization is very effective at stabilizing the hidden state dynamics in recurrent networks.
 * Empirically, we show that layer normalization can substantially reduce the training time compared
 * with previously published techniques."
 */
public class LayerNorm extends AbstractBlock {

    private float epsilon;
    private Shape normalizedShape;

    private boolean center;
    private boolean scale;
    private int[] axis;
    private Parameter gamma;
    private Parameter beta;

    LayerNorm(Builder builder) {
        epsilon = builder.epsilon;
        scale = builder.scale;
        center = builder.center;
        axis = builder.axis;

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
    }

    /**
     * Applies Layer Normalization with average and variance for each input sample across the axis
     * dimensions.
     *
     * @param input the input {@code NDArray} of shape (batchSize, inputChannel, *), * could be
     *     empty, width, (height, width), (depth, height, width)
     * @param normalizedShape dimensions to calculate average and variance from
     * @param gamma gamma weight {@code NDArray}
     * @param beta beta weight {@code NDArray}
     * @param eps a value added to the denominator for numerical stability
     * @return the output {@code NDArray} of shape (batchSize, inputChannel, *), * could be empty,
     *     width, (height, width), (depth, height, width)
     */
    public static NDList layerNorm(
            NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.layerNorm(input, normalizedShape, gamma, beta, eps);
    }

    /**
     * Creates a builder to build a {@code LayerNorm}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
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

        return layerNorm(input, normalizedShape, gammaArr, betaArr, epsilon);
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
        normalizedShape =
                axis == null
                        ? inputShapes[0].slice(1)
                        : new Shape(
                                Arrays.stream(axis)
                                        .mapToLong(dim -> inputShapes[0].get(dim))
                                        .toArray());
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Shape[] inputShapes) {
        gamma.setShape(normalizedShape);
        beta.setShape(normalizedShape);
    }

    /** {@inheritDoc} */
    @Override
    protected void saveMetadata(DataOutputStream os) throws IOException {
        saveInputShapes(os);
        os.write(normalizedShape.getEncoded());
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion != version) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        readInputShapes(is);
        normalizedShape = Shape.decode(is);
    }

    /** The Builder to construct a {@link LayerNorm}. */
    public static final class Builder {

        private float epsilon = 1E-5f;
        // private Shape normalizedShape;
        private boolean scale = true;
        private boolean center = true;
        private int[] axis;

        Builder() {}

        /**
         * List the axis over which the mean and variance will be calculated (alternative to
         * normalizedShape).
         *
         * @param axis input axis over which the mean and variance will be calculated (if null all
         *     existing dimensions)
         * @return this Builder
         */
        public Builder axis(int... axis) {
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
         * Builds a {@link LayerNorm} block.
         *
         * @return the {@link LayerNorm} block
         */
        public LayerNorm build() {
            return new LayerNorm(this);
        }
    }
}
