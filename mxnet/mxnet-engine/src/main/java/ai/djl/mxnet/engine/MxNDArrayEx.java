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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDUtils;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import java.util.Arrays;
import java.util.List;

/** {@code MxNDArrayEx} is the MXNet implementation of the {@link NDArrayEx}. */
class MxNDArrayEx implements NDArrayEx {

    private static final NDArrayIndexer INDEXER = new MxNDArrayIndexer();

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
                    "float type of normType is not supported in MXNet engine, please use integer instead");
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
                    "float type of normType is not supported in MXNet engine, please use integer instead");
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
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("beta1", beta1);
        params.addParam("beta2", beta2);
        params.addParam("epsilon", epsilon);
        params.addParam("lazy_update", lazyUpdate);

        getManager().invoke("adam_update", inputs, weights, params);
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
    public NDList embedding(
            NDList inputs,
            int numItems,
            int embeddingSize,
            boolean sparseGrad,
            DataType dataType,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("input_dim", numItems);
        params.addParam("output_dim", embeddingSize);
        params.addParam("sparse_grad", sparseGrad);
        params.setDataType(dataType);
        params.addAll(additional);

        return getManager().invoke("Embedding", inputs, params);
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
                    "the mode of batchNorm in MXNet should align with the mode of GradientCollector");
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
            NDList inputs,
            String mode,
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("p", dropRate);
        params.addParam("state_size", stateSize);
        params.addParam("num_layers", numStackedLayers);
        params.addParam("use_sequence_length", useSequenceLength);
        params.addParam("bidirectional", useBidirectional);
        params.addParam("state_outputs", stateOutputs);
        params.addParam("mode", mode);
        params.addAll(additional);
        return getManager().invoke("_npx_rnn", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList lstm(
            NDList inputs,
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs,
            double lstmStateClipMin,
            double lstmStateClipMax,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("mode", "lstm");
        params.addParam("p", dropRate);
        params.addParam("state_size", stateSize);
        params.addParam("num_layers", numStackedLayers);
        params.addParam("use_sequence_length", useSequenceLength);
        params.addParam("bidirectional", useBidirectional);
        params.addParam("state_outputs", stateOutputs);
        params.addParam("lstm_state_clip_nan", true);
        params.addParam("lstm_state_clip_min", lstmStateClipMin);
        params.addParam("lstm_state_clip_max", lstmStateClipMax);
        params.addAll(additional);

        return getManager().invoke("_npx_rnn", inputs, params);
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
    public NDArray resize(int width, int height) {
        if (array.isEmpty()) {
            throw new IllegalArgumentException("attempt to resize of an empty NDArray");
        }
        MxOpParams params = new MxOpParams();
        params.addTupleParam("size", width, height);
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

    @Override
    public NDArray randomFlipTopBottom() {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomFlipTopBottom is not supported on GPU");
        }
        return getManager().invoke("_npx__image_random_flip_top_bottom", array, null);
    }

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
    public NDArrayIndexer getIndexer() {
        return INDEXER;
    }

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDArray where(NDArray condition, NDArray other) {
        NDArray array1;
        NDArray array2;
        condition =
                (condition.getDataType() == DataType.BOOLEAN)
                        ? condition.toType(DataType.INT32, false)
                        : condition;
        if (!array.shapeEquals(other)) {
            Shape res = deriveBroadcastedShape(array.getShape(), other.getShape());
            array1 = (!res.equals(array.getShape())) ? array.broadcast(res) : array;
            array2 = (!res.equals(other.getShape())) ? other.broadcast(res) : other;
        } else {
            array1 = array;
            array2 = other;
        }
        try {
            return getManager().invoke("where", new NDArray[] {condition, array1, array2}, null);
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
        System.arraycopy(arrays.toArray(new NDArray[0]), 0, srcArray, 1, arrays.size());
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
        System.arraycopy(list.toArray(new NDArray[0]), 0, srcArray, 1, list.size());
        return getManager().invoke("_npi_concatenate", srcArray, params);
    }

    /**
     * Concats the parameters of a recurrent neural network as expected by the engine.
     *
     * @param arrays an {@link NDList} containing the the parameter arrays to be concatenated
     * @param numArgs number of inputs to be concatenated
     * @return the concatenated {@code NDArray} of parameters
     */
    public NDArray rnnParameterConcat(NDList arrays, int numArgs) {
        MxOpParams params = new MxOpParams();
        params.addParam("num_args", numArgs);
        return getManager().invoke("_npi_rnn_param_concat", arrays, params).singletonOrThrow();
    }

    /**
     * Concats the parameters of a recurrent neural network as expected by the engine.
     *
     * @param arrays an {@link NDList} containing the the parameter arrays to be concatenated
     * @param numArgs number of inputs to be concatenated
     * @param dim the dimension to be concatenated
     * @return the concatenated {@code NDArray} of parameters
     */
    public NDArray rnnParameterConcat(NDList arrays, int numArgs, int dim) {
        MxOpParams params = new MxOpParams();
        params.addParam("dim", dim);
        params.addParam("num_args", numArgs);
        return getManager().invoke("_npi_rnn_param_concat", arrays, params).singletonOrThrow();
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
