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

package ai.djl.tensorflow.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.pooling.PoolingConvention;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Stack;

public class TfNDArrayEx implements NDArrayEx {

    // private TfNDArray array;
    private TfNDManager manager;
    private Ops tf;
    private Operand<?> operand;

    /**
     * Constructs an {@code MxNDArrayEx} given a {@link NDArray}.
     *
     * @param parent the {@link NDArray} to extend
     */
    TfNDArrayEx(TfNDArray parent) {
        // this.array = parent;
        this.manager = (TfNDManager) parent.getManager();
        this.tf = manager.getTf();
        this.operand = parent.asOperand();
    }

    @Override
    public NDArray rdiv(Number n) {
        return null;
    }

    @Override
    public NDArray rdiv(NDArray b) {
        return null;
    }

    @Override
    public NDArray rdivi(Number n) {
        return null;
    }

    @Override
    public NDArray rdivi(NDArray b) {
        return null;
    }

    @Override
    public NDArray rsub(Number n) {
        return null;
    }

    @Override
    public NDArray rsub(NDArray b) {
        return null;
    }

    @Override
    public NDArray rsubi(Number n) {
        return null;
    }

    @Override
    public NDArray rsubi(NDArray b) {
        return null;
    }

    @Override
    public NDArray rmod(Number n) {
        return null;
    }

    @Override
    public NDArray rmod(NDArray b) {
        return null;
    }

    @Override
    public NDArray rmodi(Number n) {
        return null;
    }

    @Override
    public NDArray rmodi(NDArray b) {
        return null;
    }

    @Override
    public NDArray rpow(Number n) {
        return null;
    }

    @Override
    public NDArray rpowi(Number n) {
        return null;
    }

    @Override
    public NDArray relu() {
        return null;
    }

    @Override
    public NDArray sigmoid() {
        return null;
    }

    @Override
    public NDArray tanh() {
        return null;
    }

    @Override
    public NDArray softrelu() {
        return null;
    }

    @Override
    public NDArray softsign() {
        return null;
    }

    @Override
    public NDArray leakyRelu(float alpha) {
        return null;
    }

    @Override
    public NDArray elu(float alpha) {
        return null;
    }

    @Override
    public NDArray selu() {
        return null;
    }

    @Override
    public NDArray gelu() {
        return null;
    }

    @Override
    public NDArray maxPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return null;
    }

    @Override
    public NDArray globalMaxPool() {
        return null;
    }

    @Override
    public NDArray sumPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return null;
    }

    @Override
    public NDArray globalSumPool() {
        return null;
    }

    @Override
    public NDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        return null;
    }

    @Override
    public NDArray globalAvgPool() {
        return null;
    }

    @Override
    public NDArray lpPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        return null;
    }

    @Override
    public NDArray globalLpPool(int pValue) {
        return null;
    }

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

    @Override
    public void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum) {}

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

    @Override
    public NDList fullyConnected(
            NDList inputs,
            long outChannels,
            boolean flatten,
            boolean noBias,
            PairList<String, Object> additional) {
        return null;
    }

    @Override
    public NDList embedding(
            NDList inputs,
            int numItems,
            int embeddingSize,
            DataType dataType,
            PairList<String, Object> additional) {
        return null;
    }

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

    @Override
    public NDArray normalize(float[] mean, float[] std) {
        return null;
    }

    @Override
    public NDArray toTensor() {
        return null;
    }

    @SuppressWarnings("unchecked")
    @Override
    public NDArray resize(int width, int height) {
        return new TfNDArray(
                manager,
                tf.image.resizeBilinear(
                        (Operand<? extends Number>) operand,
                        tf.constant(new int[] {width, height})));
    }

    @Override
    public NDArray crop(int x, int y, int width, int height) {
        return null;
    }

    @Override
    public NDArray pick(NDArray index, int axis, boolean keepDims, String mode) {
        return null;
    }

    @Override
    public NDArray where(NDArray condition, NDArray other) {
        return null;
    }

    @Override
    public NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    @Override
    public NDArray stack(NDList arrays, int axis) {
        return stackHelper(arrays, axis);
    }

    @SuppressWarnings("unchecked")
    private <T> NDArray stackHelper(NDList arrays, int axis) {
        ArrayList<Operand<T>> operands = new ArrayList<>(arrays.size());
        for (NDArray array : arrays) {
            operands.add((Operand<T>) ((TfNDArray) array).asOperand());
        }
        return new TfNDArray(manager, tf.stack(operands, Stack.axis((long) axis)));
    }

    @Override
    public NDArray concat(NDList arrays, int axis) {
        return null;
    }

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

    @Override
    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {
        return null;
    }

    @Override
    public NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreshold,
            boolean forceSuppress,
            int nmsTopK) {
        return null;
    }

    @Override
    public NDArray getArray() {
        return null;
    }
}
