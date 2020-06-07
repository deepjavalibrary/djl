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
package ai.djl.test.mock;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.pooling.PoolingConvention;
import ai.djl.util.PairList;
import java.util.List;

public class MockNDArrayEx implements NDArrayEx {

    private MockNDArray array;

    MockNDArrayEx(MockNDArray parent) {
        this.array = parent;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(NDArray b) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpow(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpowi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray relu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sigmoid() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softrelu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softsign() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray leakyRelu(float alpha) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray elu(float alpha) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray selu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gelu() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maxPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalMaxPool() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sumPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalSumPool() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalAvgPool() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lpPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalLpPool(int pValue) {
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
            boolean sparseGrad,
            DataType dataType,
            PairList<String, Object> additional) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(NDList inputs, PairList<String, Object> additional) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList dropout(
            NDList inputs,
            float probability,
            int[] sharedAxes,
            boolean training,
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
            boolean training,
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
    public NDArray normalize(float[] mean, float[] std) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toTensor() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray resize(int width, int height) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pick(NDArray index, int axis, boolean keepDims, String mode) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray where(NDArray condition, NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays, int axis) {
        Shape newShape = new Shape(arrays.size() + 1).addAll(array.getShape());
        return new MockNDArray(null, null, newShape, null, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDList arrays, int axis) {
        return null;
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
        return null;
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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getArray() {
        return null;
    }
}
