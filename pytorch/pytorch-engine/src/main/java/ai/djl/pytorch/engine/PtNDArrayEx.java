/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.pooling.PoolingConvention;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.PairList;
import java.util.List;

/** {@code PtNDArrayEx} is the PyTorch implementation of the {@link NDArrayEx}. */
public class PtNDArrayEx implements NDArrayEx {

    private PtNDArray array;

    /**
     * Constructs an {@code PtNDArrayEx} given a {@link NDArray}.
     *
     * @param parent the {@link NDArray} to extend
     */
    PtNDArrayEx(PtNDArray parent) {
        this.array = parent;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdiv(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdiv(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdivi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdivi(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsub(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsubi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsubi(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmod(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmod(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmodi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmodi(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rpow(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rpowi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray relu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sigmoid() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray softrelu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray softsign() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray leakyRelu(float alpha) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray elu(float alpha) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray selu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray gelu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray maxPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalMaxPool() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sumPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalSumPool() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalAvgPool() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray lpPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalLpPool(int pValue) {
        return null;
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
            boolean lazyUpdate) {}

    /** {@inheritDoc} */
    @Override
    public void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum) {}

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
            boolean lazyUpdate) {}

    /** {@inheritDoc} */
    @Override
    public NDList convolution(
            NDList inputs,
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            String layout,
            boolean noBias,
            PairList<String, Object> additional) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList fullyConnected(
            NDList inputs,
            long outChannels,
            boolean flatten,
            boolean noBias,
            PairList<String, Object> additional) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(
            NDList inputs,
            int numItems,
            int embeddingSize,
            DataType dataType,
            PairList<String, Object> additional) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(NDList inputs, PairList<String, Object> additional) {
        return null;
    }

    @Override
    public NDList dropout(
            NDList inputs,
            float probability,
            int[] sharedAxes,
            PairList<String, Object> additional) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchNorm(
            NDList inputs,
            float epsilon,
            float momentum,
            int axis,
            boolean center,
            boolean scale,
            PairList<String, Object> additional) {
        return null;
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
        return null;
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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray normalize(float[] mean, float[] std) {
        return JniUtils.normalize(array, mean, std);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray toTensor() {
        return JniUtils.div(array, 255);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray resize(int width, int height) {
        if (array.isEmpty()) {
            throw new IllegalArgumentException("attempt to resize of an empty NDArray");
        }
        if (array.getDataType() != DataType.FLOAT32) {
            array = array.toType(DataType.FLOAT32, true);
        }
        return JniUtils.resize(array, new long[] {height, width}, true);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray crop(int x, int y, int width, int height) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray pick(NDArray index, int axis, boolean keepDims, String mode) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray where(NDArray condition, NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray stack(NDList arrays, int axis) {
        NDArray[] srcArray = new NDArray[arrays.size() + 1];
        srcArray[0] = array;
        System.arraycopy(arrays.toArray(new NDArray[0]), 0, srcArray, 1, arrays.size());
        return JniUtils.stack(srcArray, axis);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray concat(NDList arrays, int axis) {
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreshold,
            boolean forceSuppress,
            int nmsTopK) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray getArray() {
        return array;
    }

    private PtNDManager getManager() {
        return array.getManager();
    }
}
