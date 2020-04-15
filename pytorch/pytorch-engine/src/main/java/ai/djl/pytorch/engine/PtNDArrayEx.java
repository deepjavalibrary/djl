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
import ai.djl.ndarray.NDUtils;
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
        return rdiv(array.getManager().create(n));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdiv(NDArray b) {
        return (PtNDArray) b.div(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdivi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdivi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsub(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsub(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsubi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsubi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmod(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmod(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmodi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmodi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rpow(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rpowi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray relu() {
        return JniUtils.relu(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sigmoid() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tanh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray softrelu() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray softsign() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray leakyRelu(float alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray elu(float alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray selu() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray gelu() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray maxPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalMaxPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sumPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalSumPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalAvgPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray lpPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalLpPool(int pValue) {
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
    }

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
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList fullyConnected(
            NDList inputs,
            long outChannels,
            boolean flatten,
            boolean noBias,
            PairList<String, Object> additional) {
        NDArray result =
                JniUtils.fullyConnected(
                        (PtNDArray) inputs.get(0),
                        (PtNDArray) inputs.get(1),
                        (PtNDArray) inputs.get(2),
                        noBias);
        if (flatten) {
            long batchSize = result.getShape().get(0);
            result = result.reshape(batchSize, outChannels);
        }
        return new NDList(result);
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
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(NDList inputs, PairList<String, Object> additional) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList dropout(
            NDList inputs,
            float probability,
            int[] sharedAxes,
            PairList<String, Object> additional) {
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray resize(int width, int height) {
        NDArray result = array;
        if (result.isEmpty()) {
            throw new IllegalArgumentException("attempt to resize of an empty NDArray");
        }
        if (result.getDataType() != DataType.FLOAT32) {
            result = result.toType(DataType.FLOAT32, true);
        }
        int dim = result.getShape().dimension();
        if (dim == 3) {
            result = result.expandDims(0);
        }
        result = result.transpose(0, 3, 1, 2);
        result =
                JniUtils.upsampleBilinear2d((PtNDArray) result, new long[] {height, width}, true)
                        .transpose(0, 2, 3, 1);
        if (dim == 3) {
            result = result.squeeze(0);
        }
        return (PtNDArray) result;
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
    public PtNDArray concat(NDList list, int axis) {
        NDUtils.checkConcatInput(list);

        NDArray[] srcArray = new NDArray[list.size() + 1];
        srcArray[0] = array;
        System.arraycopy(list.toArray(new NDArray[0]), 0, srcArray, 1, list.size());
        return JniUtils.cat(srcArray, axis);
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
}
