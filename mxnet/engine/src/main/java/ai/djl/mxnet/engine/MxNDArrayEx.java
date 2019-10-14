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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.pooling.PoolingConvention;
import ai.djl.util.PairList;

class MxNDArrayEx implements NDArrayEx {

    private MxNDArray array;
    private MxNDManager manager;

    MxNDArrayEx(MxNDArray parent) {
        this.array = parent;
        this.manager = (MxNDManager) parent.getManager();
    }

    ////////////////////////////////////////
    // NDArrays
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_rdiv_scalar", array, params);
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
        manager.invoke("_rdiv_scalar", new NDList(array), new NDList(array), params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray b) {
        manager.invoke("elemwise_div", new NDList(b, array), new NDList(array), null);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return array.sub(n).negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray b) {
        return array.sub(b).negi();
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
        return manager.invoke("_npi_rmod_scalar", array, params);
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
        manager.invoke("_npi_rmod_scalar", new NDList(array), new NDList(array), params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(NDArray b) {
        manager.invoke("_npi_mod", new NDList(b, array), new NDList(array), null);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpow(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_rpower_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpowi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_npi_rpower_scalar", new NDList(array), new NDList(array), params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_maximum_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(NDArray other) {
        return manager.invoke("_npi_maximum", new NDList(array, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(NDArray other) {
        return manager.invoke("_npi_minimum", new NDList(array, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_minimum_scalar", array, params);
    }

    ////////////////////////////////////////
    // Activations
    ////////////////////////////////////////

    @Override
    public NDArray relu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "relu");
        return manager.invoke("Activation", array, params);
    }

    @Override
    public NDArray sigmoid() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "sigmoid");
        return manager.invoke("Activation", array, params);
    }

    @Override
    public NDArray tanh() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "tanh");
        return manager.invoke("Activation", array, params);
    }

    @Override
    public NDArray softrelu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "softrelu");
        return manager.invoke("Activation", array, params);
    }

    @Override
    public NDArray softsign() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "softsign");
        return manager.invoke("Activation", array, params);
    }

    @Override
    public NDArray leakyRelu(float alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "leaky");
        params.addParam("slope", alpha);
        return manager.invoke("LeakyReLU", array, params);
    }

    @Override
    public NDArray elu(float alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "elu");
        params.addParam("slope", alpha);
        return manager.invoke("LeakyReLU", array, params);
    }

    @Override
    public NDArray selu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "selu");
        return manager.invoke("LeakyReLU", array, params);
    }

    @Override
    public NDArray gelu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "gelu");
        return manager.invoke("LeakyReLU", array, params);
    }

    ////////////////////////////////////////
    // Pooling Operations
    ////////////////////////////////////////

    @Override
    public NDArray maxPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernel);
        params.add("pool_type", "max");
        params.addParam("stride", stride);
        params.addParam("pad", pad);
        if (poolingConvention != null) {
            params.add("pooling_convention", poolingConvention.name().toLowerCase());
        }
        return pool(params);
    }

    @Override
    public NDArray globalMaxPool() {
        MxOpParams params = new MxOpParams();
        params.add("pool_type", "max");
        params.addParam("global_pool", true);
        return pool(params);
    }

    @Override
    public NDArray sumPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernel);
        params.add("pool_type", "sum");
        params.addParam("stride", stride);
        params.addParam("pad", pad);
        if (poolingConvention != null) {
            params.add("pooling_convention", poolingConvention.name().toLowerCase());
        }
        return pool(params);
    }

    @Override
    public NDArray globalSumPool() {
        MxOpParams params = new MxOpParams();
        params.add("pool_type", "sum");
        params.addParam("global_pool", true);
        return pool(params);
    }

    @Override
    public NDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernel);
        params.add("pool_type", "avg");
        params.addParam("stride", stride);
        params.addParam("pad", pad);
        params.addParam("count_include_pad", countIncludePad);
        if (poolingConvention != null) {
            params.add("pooling_convention", poolingConvention.name().toLowerCase());
        }
        return pool(params);
    }

    @Override
    public NDArray globalAvgPool() {
        MxOpParams params = new MxOpParams();
        params.add("pool_type", "avg");
        params.addParam("global_pool", true);
        return pool(params);
    }

    @Override
    public NDArray lpPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernel);
        params.add("pool_type", "lp");
        params.addParam("stride", stride);
        params.addParam("pad", pad);
        params.addParam("p_value", pValue);
        if (poolingConvention != null) {
            params.add("pooling_convention", poolingConvention.name().toLowerCase());
        }
        return pool(params);
    }

    @Override
    public NDArray globalLpPool(int pValue) {
        MxOpParams params = new MxOpParams();
        params.add("pool_type", "lp");
        params.addParam("p_value", pValue);
        params.addParam("global_pool", true);
        return pool(params);
    }

    private NDArray pool(MxOpParams params) {
        return manager.invoke("Pooling", getArray(), params);
    }

    ////////////////////////////////////////
    // Optimizer
    ////////////////////////////////////////

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

        manager.invoke("adam_update", inputs, weights, params);
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

        if (momentum != 0) {
            params.addParam("momentum", momentum);
            manager.invoke("nag_mom_update", inputs, weights, params);
        } else {
            manager.invoke("sgd_update", inputs, weights, params);
        }
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
            manager.invoke("sgd_mom_update", inputs, weights, params);
        } else {
            manager.invoke("sgd_update", inputs, weights, params);
        }
    }

    ////////////////////////////////////////
    // Neural network
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDList convolution(
            NDList inputs,
            Shape kernel,
            Shape stride,
            Shape pad,
            int numFilters,
            int numGroups,
            String layout,
            boolean noBias,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernel);
        params.addParam("stride", stride);
        params.addParam("pad", pad);
        params.addParam("num_filter", numFilters);
        params.addParam("num_group", numGroups);
        params.add("layout", layout);
        params.add("no_bias", noBias);
        params.addAll(additional);

        return manager.invoke("Convolution", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList fullyConnected(
            NDList inputs,
            long outChannels,
            boolean flatten,
            boolean noBias,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("num_hidden", outChannels);
        params.addParam("flatten", flatten);
        params.addParam("no_bias", noBias);
        params.addAll(additional);

        return manager.invoke("FullyConnected", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(
            NDList inputs,
            int numItems,
            int embeddingSize,
            DataType dataType,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("input_dim", numItems);
        params.addParam("output_dim", embeddingSize);
        params.addParam("sparse_grad", true);
        params.setDataType(dataType);
        params.addAll(additional);

        return manager.invoke("Embedding", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(NDList inputs, PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "prelu");
        params.addAll(additional);

        return manager.invoke("LeakyReLU", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList dropout(
            NDList inputs,
            float probability,
            int[] sharedAxes,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("p", probability);
        params.addTupleParam("axes", sharedAxes);
        params.addAll(additional);

        return manager.invoke("Dropout", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchNorm(
            NDList inputs,
            float epsilon,
            float momentum,
            int axis,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("eps", epsilon);
        params.addParam("momentum", momentum);
        params.addParam("axis", axis);
        params.addAll(additional);

        return manager.invoke("BatchNorm", inputs, params);
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

        return manager.invoke("RNN", inputs, params);
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
            double lstmStateClipMin,
            double lstmStateClipMax,
            PairList<String, Object> additional) {
        MxOpParams params = new MxOpParams();
        params.addParam("mode", mode);
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

        return manager.invoke("RNN", inputs, params);
    }

    ////////////////////////////////////////
    // Image and CV
    ////////////////////////////////////////

    @Override
    public NDArray normalize(float[] mean, float[] std) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("mean", mean);
        params.addTupleParam("std", std);
        return manager.invoke("_npx__image_normalize", array, params);
    }

    @Override
    public NDArray toTensor() {
        return manager.invoke("_npx__image_to_tensor", array, null);
    }

    @Override
    public NDArray resize(int height, int width) {
        if (array.isEmpty()) {
            throw new IllegalArgumentException("attempt to resize of an empty NDArray");
        }
        MxOpParams params = new MxOpParams();
        params.addTupleParam("size", width, height);
        return manager.invoke("_npx__image_resize", array, params);
    }

    @Override
    public NDArray crop(int y, int x, int height, int width) {
        MxOpParams params = new MxOpParams();
        params.add("x", x);
        params.add("y", y);
        params.add("width", width);
        params.add("height", height);
        return manager.invoke("_npx__image_crop", array, params);
    }

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDArray pick(NDArray index, int axis, boolean keepDims, String mode) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.addParam("keepdims", keepDims);
        params.add("mode", mode);
        return manager.invoke("pick", new NDList(array, index), params).head();
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
        return manager.invoke("MultiBoxTarget", inputs, parameters);
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
        return manager.invoke("MultiBoxDetection", inputs, parameters);
    }

    @Override
    public NDArray getArray() {
        return array;
    }
}
