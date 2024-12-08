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
package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDUtils;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;
import ai.djl.util.Preconditions;

import java.util.Arrays;
import java.util.List;

/** {@code MxNDArrayEx} is the MXNet implementation of the {@link NDArrayEx}. */
@SuppressWarnings("dangling-doc-comments")
class MxNDArrayEx implements NDArrayEx {

    private MxNDArray array;

    /**
     * Constructs an {@code MxNDArrayEx} given a {@link NDArray}.
     *
     * @param parent the {@link NDArray} to extend
     */
    MxNDArrayEx(MxNDArray parent) {
        this.array = parent;
    }

    // TODO only used to calculate zero-dim numpy shape
    // remove it once MXNet have all the np op that we support
    private Shape deriveBroadcastedShape(Shape lhs, Shape rhs) {
        long[] result = new long[Math.max(lhs.dimension(), rhs.dimension())];
        long lDiff = result.length - lhs.dimension();
        long rDiff = result.length - rhs.dimension();
        for (int i = 0; i < result.length; i++) {
            long l = 1;
            long r = 1;
            if (i >= lDiff) {
                l = lhs.get(Math.toIntExact(i - lDiff));
            }
            if (i >= rDiff) {
                r = rhs.get(Math.toIntExact(i - rDiff));
            }
            if (l != r) {
                if (l != 1 && r != 1) {
                    throw new IllegalArgumentException(
                            "operands could not be broadcast together with shapes "
                                    + lhs
                                    + " "
                                    + rhs);
                }
                result[i] = (l == 1) ? r : l;
            } else {
                result[i] = l;
            }
        }
        return new Shape(result);
    }

    ////////////////////////////////////////
    // NDArrays
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_rdiv_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray b) {
        return b.div(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke("_rdiv_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray b) {
        b = getManager().from(b);
        getManager().invoke("elemwise_div", new NDArray[] {b, array}, new NDArray[] {array}, null);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return array.sub(n).neg();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray b) {
        return array.sub(b).neg();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
        return array.subi(n).negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray b) {
        return array.subi(b).negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_rmod_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(NDArray b) {
        return b.mod(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager()
                .invoke("_npi_rmod_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(NDArray b) {
        b = getManager().from(b);
        getManager().invoke("_npi_mod", new NDArray[] {b, array}, new NDArray[] {array}, null);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpow(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_rpower_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpowi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager()
                .invoke("_npi_rpower_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
        return array;
    }

    ////////////////////////////////////////
    // Activations
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDArray relu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "relu");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sigmoid() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "sigmoid");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "tanh");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softPlus() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "softrelu");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softSign() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "softsign");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray leakyRelu(float alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "leaky");
        params.addParam("slope", alpha);
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray elu(float alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "elu");
        params.addParam("slope", alpha);
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray selu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "selu");
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gelu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "gelu");
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    ////////////////////////////////////////
    // Pooling Operations
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "max");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");
        return getManager().invoke("_npx_pooling", getArray(), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalMaxPool() {
        MxOpParams params = new MxOpParams();
        params.add("kernel", getGlobalPoolingShapes(1));
        params.add("pad", getGlobalPoolingShapes(0));
        params.add("pool_type", "max");
        params.addParam("global_pool", true);
        try (NDArray temp = getManager().invoke("_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "avg");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");
        params.addParam("count_include_pad", countIncludePad);
        return getManager().invoke("_npx_pooling", getArray(), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalAvgPool() {
        MxOpParams params = new MxOpParams();
        params.add("kernel", getGlobalPoolingShapes(1));
        params.add("pad", getGlobalPoolingShapes(0));
        params.add("pool_type", "avg");
        params.addParam("global_pool", true);
        try (NDArray temp = getManager().invoke("_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        if (((int) normType) != normType) {
            throw new IllegalArgumentException(
                    "float type of normType is not supported in MXNet engine, please use integer"
                            + " instead");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("p_value", (int) normType);
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "lp");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");

        return getManager().invoke("_npx_pooling", getArray(), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalLpPool(float normType) {
        if (((int) normType) != normType) {
            throw new IllegalArgumentException(
                    "float type of normType is not supported in MXNet engine, please use integer"
                            + " instead");
        }
        MxOpParams params = new MxOpParams();
        params.add("pool_type", "lp");
        params.addParam("p_value", (int) normType);
        params.addParam("global_pool", true);
        try (NDArray temp = getManager().invoke("_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    ////////////////////////////////////////
    // Optimizer
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public void adadeltaUpdate(
            NDList inputs,
            NDList weights,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float epsilon) {
        NDArray weight = inputs.get(0);
        NDArray grad = inputs.get(1);
        NDArray s = inputs.get(2);
        NDArray delta = inputs.get(3);

        // create a baseManager to close all intermediate NDArrays
        try (NDManager subManager = NDManager.newBaseManager()) {
            subManager.tempAttachAll(inputs, weights);

            // Preprocess Gradient
            grad.muli(rescaleGrad);
            if (clipGrad > 0) {
                grad = grad.clip(-clipGrad, clipGrad);
            }
            grad.addi(weight.mul(weightDecay));

            // Update s, g, and delta
            s.muli(rho).addi(grad.square().mul(1 - rho));
            NDArray g = delta.add(epsilon).sqrt().div(s.add(epsilon).sqrt()).mul(grad);
            delta.muli(rho).addi(g.square().mul(1 - rho));

            // Update weight
            weight.subi(g);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void adagradUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float epsilon) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("epsilon", epsilon);

        getManager().invoke("adagrad_update", inputs, weights, params);
    }

    /** {@inheritDoc} */
    @Override
    public void adamUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float learningRateBiasCorrection,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate,
            boolean adamw) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRateBiasCorrection);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("beta1", beta1);
        params.addParam("beta2", beta2);
        params.addParam("epsilon", epsilon);

        if (!adamw) {
            params.addParam("wd", weightDecay);
            params.addParam("rescale_grad", rescaleGrad);
            params.addParam("lazy_update", lazyUpdate);
            getManager().invoke("adam_update", inputs, weights, params);
        } else {
            // https://github.com/apache/mxnet/blob/7d602e3b2382eb501fdeb94c4d97e652a723af11/src/operator/contrib/adamw.cc#L80-L121
            // https://github.com/apache/mxnet/blob/7d602e3b2382eb501fdeb94c4d97e652a723af11/src/operator/contrib/adamw-inl.h#L172-L207
            inputs.add(inputs.getManager().create(rescaleGrad));
            params.addParam("eta", 1.0f);
            params.addParam("wd", weightDecay * learningRate);
            getManager().invoke("_adamw_update", inputs, weights, params);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void rmspropUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float gamma1,
            float gamma2,
            float epsilon,
            boolean centered) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("gamma1", gamma1);
        params.addParam("epsilon", epsilon);

        if (!centered) {
            getManager().invoke("rmsprop_update", inputs, weights, params);
        } else {
            params.addParam("gamma2", gamma2);

            getManager().invoke("rmspropalex_update", inputs, weights, params);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);
        params.addParam("momentum", momentum);
        getManager().invoke("nag_mom_update", inputs, weights, params);
    }

    /** {@inheritDoc} */
    @Override
    public void sgdUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum,
            boolean lazyUpdate) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);
        params.addParam("lazy_update", lazyUpdate);

        if (momentum != 0) {
            params.addParam("momentum", momentum);
            getManager().invoke("sgd_mom_update", inputs, weights, params);
        } else {
            getManager().invoke("sgd_update", inputs, weights, params);
        }
    }

    ////////////////////////////////////////
    // Neural network
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDList convolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", weight.getShape().slice(2));
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.addParam("dilate", dilation);
        params.addParam("num_group", groups);
        params.addParam("num_filter", weight.getShape().get(0));

        NDList inputs = new NDList(input, weight);
        if (bias != null) {
            params.add("no_bias", false);
            inputs.add(bias);
        } else {
            params.add("no_bias", true);
        }

        return getManager().invoke("_npx_convolution", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList deconvolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", weight.getShape().slice(2));
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.addParam("adj", outPadding);
        params.addParam("dilate", dilation);
        params.addParam("num_group", groups);
        params.addParam("num_filter", weight.getShape().get(0));

        NDList inputs = new NDList(input, weight);
        if (bias != null) {
            params.add("no_bias", false);
            inputs.add(bias);
        } else {
            params.add("no_bias", true);
        }

        return getManager().invoke("_npx_deconvolution", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList linear(NDArray input, NDArray weight, NDArray bias) {
        MxOpParams params = new MxOpParams();
        params.addParam("num_hidden", weight.size(0));
        params.addParam("flatten", false);
        params.addParam("no_bias", bias == null);
        NDList inputs = new NDList(input, weight);
        if (bias != null) {
            inputs.add(bias);
        }

        return getManager().invoke("_npx_fully_connected", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparse) {
        if (!sparse.equals(SparseFormat.DENSE) && !sparse.equals(SparseFormat.ROW_SPARSE)) {
            throw new IllegalArgumentException("MXNet only supports row sparse");
        }
        MxOpParams params = new MxOpParams();
        long inputDim = weight.getShape().get(0);
        long outputDim = weight.getShape().get(1);
        params.addParam("input_dim", inputDim);
        params.addParam("output_dim", outputDim);
        params.addParam("sparse_grad", sparse.getValue());
        return getManager().invoke("_npx_embedding", new NDList(input, weight), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(NDArray input, NDArray alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "prelu");
        return getManager().invoke("_npx_leaky_relu", new NDList(input, alpha), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList dropout(NDArray input, float rate, boolean training) {
        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of dropout in MXNet should align with the mode of GradientCollector");
        }

        MxOpParams params = new MxOpParams();
        params.addParam("p", rate);

        return getManager().invoke("_npx_dropout", new NDList(input), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList layerNorm(
            NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps) {

        MxOpParams params = new MxOpParams();
        params.addParam("axis", -1);
        params.addParam("eps", eps);

        NDArray reshapedInput =
                input.reshape(
                        input.getShape()
                                .slice(
                                        0,
                                        Math.toIntExact(
                                                input.getShape().dimension()
                                                        - normalizedShape.dimension()))
                                .add(normalizedShape.size()));

        // Cause of gamma and betta attached to model manager we must attach them to input NDManager
        // to avoid memory leak.
        final NDArray reshapedGamma = gamma.reshape(normalizedShape.size());
        final NDArray reshapedBeta = beta.reshape(normalizedShape.size());
        final NDManager inputManager = input.getManager();
        reshapedBeta.attach(inputManager);
        reshapedGamma.attach(inputManager);

        return new NDList(
                getManager()
                        .invoke(
                                "_npx_layer_norm",
                                new NDList(reshapedInput, reshapedGamma, reshapedBeta),
                                params)
                        .get(0)
                        .reshape(input.getShape()));
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchNorm(
            NDArray input,
            NDArray runningMean,
            NDArray runningVar,
            NDArray gamma,
            NDArray beta,
            int axis,
            float momentum,
            float eps,
            boolean training) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.addParam("fix_gamma", gamma == null);
        params.addParam("eps", eps);
        params.addParam("momentum", momentum);

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of batchNorm in MXNet should align with the mode of"
                            + " GradientCollector");
        }

        return getManager()
                .invoke(
                        "_npx_batch_norm",
                        new NDList(input, gamma, beta, runningMean, runningVar),
                        params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList rnn(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            RNN.Activation activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
        Preconditions.checkArgument(
                params.size() == numParams,
                "The size of Params is incorrect expect "
                        + numParams
                        + " parameters but got "
                        + params.size());

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of rnn in MXNet should align with the mode of GradientCollector");
        }

        if (batchFirst) {
            input = input.swapAxes(0, 1);
        }

        MxOpParams opParams = new MxOpParams();
        opParams.addParam("p", dropRate);
        opParams.addParam("state_size", state.getShape().tail());
        opParams.addParam("num_layers", numLayers);
        opParams.addParam("bidirectional", bidirectional);
        opParams.addParam("state_outputs", true);
        opParams.addParam("mode", activation == RNN.Activation.TANH ? "rnn_tanh" : "rnn_relu");

        NDList inputs = new NDList();
        inputs.add(input);

        try (NDList temp = new NDList()) {
            for (NDArray param : params) {
                temp.add(param.flatten());
            }
            NDArray tempParam = NDArrays.concat(temp);
            tempParam.attach(input.getManager());
            inputs.add(tempParam);
        }

        inputs.add(state);

        if (!batchFirst) {
            return getManager().invoke("_npx_rnn", inputs, opParams);
        }

        NDList result = getManager().invoke("_npx_rnn", inputs, opParams);
        try (NDArray temp = result.head()) {
            return new NDList(temp.swapAxes(0, 1), result.get(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList gru(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
        Preconditions.checkArgument(
                params.size() == numParams,
                "The size of Params is incorrect expect "
                        + numParams
                        + " parameters but got "
                        + params.size());

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of gru in MXNet should align with the mode of GradientCollector");
        }

        if (batchFirst) {
            input = input.swapAxes(0, 1);
        }

        MxOpParams opParams = new MxOpParams();
        opParams.addParam("p", dropRate);
        opParams.addParam("state_size", state.getShape().tail());
        opParams.addParam("num_layers", numLayers);
        opParams.addParam("bidirectional", bidirectional);
        opParams.addParam("state_outputs", true);
        opParams.addParam("mode", "gru");

        NDList inputs = new NDList();
        inputs.add(input);

        try (NDList temp = new NDList()) {
            for (NDArray param : params) {
                temp.add(param.flatten());
            }
            NDArray tempParam = NDArrays.concat(temp);
            tempParam.attach(input.getManager());
            inputs.add(tempParam);
        }

        inputs.add(state);

        if (!batchFirst) {
            return getManager().invoke("_npx_rnn", inputs, opParams);
        }

        NDList result = getManager().invoke("_npx_rnn", inputs, opParams);
        try (NDArray temp = result.head()) {
            return new NDList(temp.swapAxes(0, 1), result.get(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList lstm(
            NDArray input,
            NDList states,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        if (!hasBiases) {
            throw new UnsupportedOperationException(
                    "Setting hasBias to be false is not supported on MXNet engine.");
        }
        int numParams = numLayers * 4 * (bidirectional ? 2 : 1);
        Preconditions.checkArgument(
                params.size() == numParams,
                "The size of Params is incorrect expect "
                        + numParams
                        + " parameters but got "
                        + params.size());

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of lstm in MXNet should align with the mode of GradientCollector");
        }

        if (batchFirst) {
            input = input.swapAxes(0, 1);
        }

        MxOpParams opParams = new MxOpParams();
        opParams.addParam("mode", "lstm");
        opParams.addParam("p", dropRate);
        opParams.addParam("state_size", states.head().getShape().tail());
        opParams.addParam("state_outputs", true);
        opParams.addParam("num_layers", numLayers);
        opParams.addParam("bidirectional", bidirectional);
        opParams.addParam("lstm_state_clip_nan", true);

        NDList inputs = new NDList();
        inputs.add(input);
        try (NDList temp = new NDList()) {
            for (NDArray param : params) {
                temp.add(param.flatten());
            }
            NDArray tempParam = NDArrays.concat(temp);
            tempParam.attach(input.getManager());
            inputs.add(tempParam);
        }
        inputs.addAll(states);

        if (!batchFirst) {
            return getManager().invoke("_npx_rnn", inputs, opParams);
        }

        NDList result = getManager().invoke("_npx_rnn", inputs, opParams);
        try (NDArray temp = result.head()) {
            return new NDList(temp.swapAxes(0, 1), result.get(1), result.get(2));
        }
    }

    ////////////////////////////////////////
    // Image and CV
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDArray normalize(float[] mean, float[] std) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("mean", mean);
        params.addTupleParam("std", std);
        return getManager().invoke("_npx__image_normalize", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toTensor() {
        return getManager().invoke("_npx__image_to_tensor", array, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray interpolation(long[] size, int mode, boolean alignCorners) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray resize(int width, int height, int interpolation) {
        if (array.isEmpty()) {
            throw new IllegalArgumentException("attempt to resize of an empty NDArray");
        }
        MxOpParams params = new MxOpParams();
        params.addTupleParam("size", width, height);
        params.addParam("interp", interpolation);
        return getManager().invoke("_npx__image_resize", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray crop(int x, int y, int width, int height) {
        MxOpParams params = new MxOpParams();
        params.add("x", x);
        params.add("y", y);
        params.add("width", width);
        params.add("height", height);
        return getManager().invoke("_npx__image_crop", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomFlipLeftRight() {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomFlipLeftRight is not supported on GPU");
        }
        return getManager().invoke("_npx__image_random_flip_left_right", array, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomFlipTopBottom() {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomFlipTopBottom is not supported on GPU");
        }
        return getManager().invoke("_npx__image_random_flip_top_bottom", array, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomBrightness(float brightness) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomBrightness is not supported on GPU");
        }
        MxOpParams params = new MxOpParams();
        float min = Math.max(0, 1 - brightness);
        float max = 1 + brightness;
        params.addParam("min_factor", min);
        params.addParam("max_factor", max);
        return getManager().invoke("_npx__image_random_brightness", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomHue(float hue) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomHue is not supported on GPU");
        }
        MxOpParams params = new MxOpParams();
        float min = Math.max(0, 1 - hue);
        float max = 1 + hue;
        params.addParam("min_factor", min);
        params.addParam("max_factor", max);
        return getManager().invoke("_npx__image_random_hue", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomColorJitter(
            float brightness, float contrast, float saturation, float hue) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomColorJitter is not supported on GPU");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("brightness", brightness);
        params.addParam("contrast", contrast);
        params.addParam("saturation", saturation);
        params.addParam("hue", hue);
        return getManager().invoke("_npx__image_random_color_jitter", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayIndexer getIndexer(NDManager manager) {
        return new MxNDArrayIndexer((MxNDManager) manager);
    }

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.UseTryWithResources")
    public NDArray where(NDArray condition, NDArray other) {
        NDArray array1;
        NDArray array2;
        condition =
                (condition.getDataType() == DataType.BOOLEAN)
                        ? condition.toType(DataType.INT32, false)
                        : condition;
        if (array.getDataType() != other.getDataType()) {
            throw new IllegalArgumentException(
                    "DataType mismatch, required "
                            + array.getDataType()
                            + " actual "
                            + other.getDataType());
        }
        if (!array.shapeEquals(other)) {
            Shape res = deriveBroadcastedShape(array.getShape(), other.getShape());
            array1 = (!res.equals(array.getShape())) ? array.broadcast(res) : array;
            array2 = (!res.equals(other.getShape())) ? other.broadcast(res) : other;
        } else {
            array1 = array;
            array2 = other;
        }
        try {
            MxNDManager manager = getManager();
            return manager.invoke(
                    "where",
                    new NDArray[] {manager.from(condition), array1, manager.from(array2)},
                    null);
        } finally {
            if (array1 != array) {
                array1.close();
            }
            if (array2 != other) {
                array2.close();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays, int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        NDArray[] srcArray = new NDArray[arrays.size() + 1];
        srcArray[0] = array;
        NDManager manager = array.getManager();
        int i = 1;
        for (NDArray arr : arrays) {
            srcArray[i++] = manager.from(arr);
        }
        return getManager().invoke("_npi_stack", srcArray, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDList list, int axis) {
        NDUtils.checkConcatInput(list);

        MxOpParams params = new MxOpParams();
        // MXNet backend use dim as argument name
        params.addParam("axis", axis);
        NDArray[] srcArray = new NDArray[list.size() + 1];
        srcArray[0] = array;
        NDManager manager = array.getManager();
        int i = 1;
        for (NDArray arr : list) {
            srcArray[i++] = manager.from(arr);
        }
        return getManager().invoke("_npi_concatenate", srcArray, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxTarget(
            NDList inputs,
            float iouThreshold,
            float ignoreLabel,
            float negativeMiningRatio,
            float negativeMiningThreshold,
            int minNegativeSamples) {
        MxOpParams parameters = new MxOpParams();
        parameters.add("minimum_negative_samples", minNegativeSamples);
        parameters.add("overlap_threshold", iouThreshold);
        parameters.add("ignore_label", ignoreLabel);
        parameters.add("negative_mining_ratio", negativeMiningRatio);
        parameters.add("negative_mining_thresh", negativeMiningThreshold);
        return getManager().invoke("MultiBoxTarget", inputs, parameters);
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {
        MxOpParams parameters = new MxOpParams();
        parameters.add("sizes", sizes);
        parameters.add("ratios", ratios);
        parameters.add("steps", steps);
        parameters.add("offsets", offsets);
        parameters.add("clip", clip);
        return getManager().invoke("MultiBoxPrior", new NDList(array), parameters);
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreashold,
            boolean forceSuppress,
            int nmsTopK) {
        MxOpParams parameters = new MxOpParams();
        parameters.add("clip", clip);
        parameters.add("threshold", threshold);
        parameters.add("background_id", backgroundId);
        parameters.add("nms_threshold", nmsThreashold);
        parameters.add("force_suppress", forceSuppress);
        parameters.add("nms_topk", nmsTopK);
        return getManager().invoke("MultiBoxDetection", inputs, parameters);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getArray() {
        return array;
    }

    private MxNDManager getManager() {
        return array.getManager();
    }

    private int getGlobalPoolingDim() {
        int poolDim = getArray().getShape().dimension() - 2;
        if (poolDim < 1 || poolDim > 3) {
            throw new IllegalStateException(
                    "GlobalPooling only support"
                            + "1 to 3 Dimensions, "
                            + poolDim
                            + "D is not supported.");
        }
        return poolDim;
    }

    private Shape getGlobalPoolingShapes(long fillValue) {
        // determine pooling dimension according to input
        // input dimension minus 2 (batch and channel dim)
        int poolDim = getGlobalPoolingDim();
        long[] shape = new long[poolDim];
        Arrays.fill(shape, fillValue);
        return new Shape(shape);
    }
}
