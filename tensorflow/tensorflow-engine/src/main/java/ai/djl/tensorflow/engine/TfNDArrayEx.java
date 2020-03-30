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
import ai.djl.ndarray.NDUtils;
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
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

public class TfNDArrayEx implements NDArrayEx {

    private TfNDArray array;
    private TfNDManager manager;
    private Ops tf;
    private Operand<?> operand;

    /**
     * Constructs an {@code MxNDArrayEx} given a {@link NDArray}.
     *
     * @param parent the {@link NDArray} to extend
     */
    TfNDArrayEx(TfNDArray parent) {
        this.array = parent;
        this.manager = (TfNDManager) parent.getManager();
        this.tf = manager.getTf();
        this.operand = parent.asOperand();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        return rdiv(manager.create(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray b) {
        return b.div(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        return rdivi(manager.create(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray b) {
        return b.divi(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return rsub(manager.create(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray b) {
        return b.sub(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
        return rsubi(manager.create(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray b) {
        return b.subi(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(Number n) {
        return rmod(manager.create(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(NDArray b) {
        return b.mod(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(Number n) {
        return rmodi(manager.create(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(NDArray b) {
        return b.modi(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpow(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpowi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray relu() {
        return new TfNDArray(manager, tf.nn.relu(array.asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sigmoid() {
        return new TfNDArray(manager, tf.math.sigmoid(array.asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return array.tanh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softrelu() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softsign() {
        return new TfNDArray(manager, tf.nn.softsign(array.asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray leakyRelu(float alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray elu(float alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray selu() {
        return new TfNDArray(manager, tf.nn.selu(array.asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gelu() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maxPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalMaxPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sumPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalSumPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalAvgPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lpPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalLpPool(int pValue) {
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
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(
            NDList inputs,
            int numItems,
            int embeddingSize,
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
    @SuppressWarnings("unchecked")
    @Override
    public NDArray resize(int width, int height) {
        if (array.getShape().dimension() == 3) {
            return new TfNDArray(
                    manager,
                    tf.squeeze(
                            tf.image.resizeBilinear(
                                    ((TfNDArray) array.expandDims(0)).asOperand(),
                                    tf.constant(new int[] {width, height}))));
        }
        return new TfNDArray(
                manager,
                tf.image.resizeBilinear(
                        (Operand<? extends TNumber>) operand,
                        tf.constant(new int[] {width, height})));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pick(NDArray index, int axis, boolean keepDims, String mode) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray where(NDArray condition, NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays, int axis) {
        return stackHelper(arrays, axis);
    }

    private <T extends TType> NDArray stackHelper(NDList arrays, int axis) {
        ArrayList<Operand<T>> operands = new ArrayList<>(arrays.size() + 1);
        operands.add(array.asOperand());
        for (NDArray ndArray : arrays) {
            operands.add(((TfNDArray) ndArray).asOperand());
        }
        return new TfNDArray(manager, tf.stack(operands, Stack.axis((long) axis)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDList arrays, int axis) {
        NDUtils.checkConcatInput(arrays);
        return concatHelper(arrays, axis);
    }

    private <T extends TType> NDArray concatHelper(NDList arrays, int axis) {
        ArrayList<Operand<T>> operands = new ArrayList<>(arrays.size() + 1);
        operands.add(array.asOperand());
        for (NDArray ndArray : arrays) {
            operands.add(((TfNDArray) ndArray).asOperand());
        }
        return new TfNDArray(manager, tf.concat(operands, tf.constant(axis)));
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
    public NDArray getArray() {
        return array;
    }
}
