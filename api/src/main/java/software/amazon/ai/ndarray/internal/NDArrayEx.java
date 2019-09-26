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
package software.amazon.ai.ndarray.internal;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.pooling.PoolingConvention;
import software.amazon.ai.training.Activation;
import software.amazon.ai.util.PairList;

/** An internal interface that encapsulate engine specific operator methods. */
public interface NDArrayEx {

    ////////////////////////////////////////
    // NDArrays
    ////////////////////////////////////////

    /**
     * Reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param n Value to use for reverse division
     * @return copy of array after applying reverse division
     */
    NDArray rdiv(Number n);

    /**
     * Reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param b ndarray to use for reverse division
     * @return copy of array after applying reverse division
     */
    NDArray rdiv(NDArray b);

    /**
     * In place reverse division - i.e., (n / thisArrayValues).
     *
     * @param n Value to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rdivi(Number n);

    /**
     * In place reverse division - i.e., (n / thisArrayValues).
     *
     * @param b ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rdivi(NDArray b);

    /**
     * Reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param n Value to use for reverse subtraction
     * @return copy of array after reverse subtraction
     */
    NDArray rsub(Number n);

    /**
     * Reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param b ndarray to use for reverse subtraction
     * @return copy of array after reverse subtraction
     */
    NDArray rsub(NDArray b);

    /**
     * Reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param n Value to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    NDArray rsubi(Number n);

    /**
     * Reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param b ndarray to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    NDArray rsubi(NDArray b);

    /**
     * Reverse remainder of division with a scalar.
     *
     * @param n Value to use for reverse division
     * @return copy of array after applying reverse division
     */
    NDArray rmod(Number n);

    /**
     * Reverse remainder of division.
     *
     * @param b ndarray to use for reverse division
     * @return copy of array after applying reverse division
     */
    NDArray rmod(NDArray b);

    /**
     * In place reverse remainder of division with a scalar.
     *
     * @param n Value to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rmodi(Number n);

    /**
     * In place reverse remainder of division.
     *
     * @param b ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rmodi(NDArray b);

    /**
     * Reverse the power of each element being raised in the {@code NDArray}.
     *
     * @param n Value to use for reverse power
     * @return copy of array after applying reverse power
     */
    NDArray rpow(Number n);

    /**
     * In place reverse the power of each element being raised in the {@code NDArray}.
     *
     * @param n Value to use for reverse power
     * @return copy of array after applying reverse power
     */
    NDArray rpowi(Number n);

    /**
     * Returns element-wise maximum of the input arrays with broadcasting.
     *
     * @param n Value to be compared
     * @return the maximum of two {@code NDArray}.
     */
    NDArray max(Number n);

    /**
     * Returns element-wise maximum of the input arrays with broadcasting.
     *
     * @param other the arrays holding the elements to be compared. They must have the same shape,
     *     or shapes that can be broadcast to a single shape.
     * @return the maximum of two {@code NDArray}.
     */
    NDArray max(NDArray other);

    /**
     * Returns element-wise minimum of the input arrays with broadcasting.
     *
     * @param n Value to be compared
     * @return the minimum of two {@code NDArray}.
     */
    NDArray min(Number n);

    /**
     * Returns element-wise minimum of the input arrays with broadcasting.
     *
     * @param other the arrays holding the elements to be compared. They must have the same shape,
     *     or shapes that can be broadcast to a single shape.
     * @return the minimum of two {@code NDArray}.
     */
    NDArray min(NDArray other);

    ////////////////////////////////////////
    // Activations
    ////////////////////////////////////////

    /**
     * Computes rectified linear activation.
     *
     * @return copy of array after applying relu
     */
    NDArray relu();

    NDArray sigmoid();

    NDArray tanh();

    NDArray softrelu();

    NDArray softsign();

    NDArray leakyRelu(float alpha);

    NDArray elu(float alpha);

    NDArray selu();

    NDArray gelu();

    default NDArray swish(float beta) {
        return Activation.sigmoid(getArray().mul(beta)).mul(getArray());
    }

    ////////////////////////////////////////
    // Pooling Operations
    ////////////////////////////////////////

    NDArray maxPool(Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention);

    /**
     * Normalize a NDArray of shape (C x H x W) or (N x C x H x W) with mean and standard deviation.
     *
     * <p>Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,
     * this transform normalizes each channel of the input tensor with: output[i] = (input[i] - m\
     * :sub:`i`\ ) / s\ :sub:`i`
     *
     * @param mean mean value for each channel
     * @param std standard deviation for each channel
     * @return the result of normalization
     */
    NDArray normalize(float[] mean, float[] std);

    NDArray globalMaxPool(Shape stride, Shape pad, PoolingConvention poolingConvention);

    NDArray sumPool(Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention);

    NDArray globalSumPool(Shape stride, Shape pad, PoolingConvention poolingConvention);

    NDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad);

    NDArray globalAvgPool(
            Shape stride, Shape pad, PoolingConvention poolingConvention, boolean countIncludePad);

    NDArray lpPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention, int pValue);

    NDArray globalLpPool(Shape stride, Shape pad, PoolingConvention poolingConvention, int pValue);

    ////////////////////////////////////////
    // Optimizer
    ////////////////////////////////////////

    void adamUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate);

    void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum);

    void sgdUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum,
            boolean lazyUpdate);

    ////////////////////////////////////////
    // Neural network
    ////////////////////////////////////////

    NDList convolution(
            NDList inputs,
            Shape kernel,
            Shape stride,
            Shape pad,
            int numFilters,
            int numGroups,
            String layout,
            boolean noBias,
            PairList<String, Object> additional);

    NDList fullyConnected(
            NDList inputs,
            long outChannels,
            boolean flatten,
            boolean noBias,
            PairList<String, Object> additional);

    NDList embedding(
            NDList inputs,
            int numItems,
            int embeddingSize,
            DataType dataType,
            PairList<String, Object> additional);

    NDList prelu(NDList inputs, PairList<String, Object> additional);

    NDList dropout(
            NDList inputs,
            float probability,
            int[] sharedAxes,
            PairList<String, Object> additional);

    NDList batchNorm(
            NDList inputs,
            float epsilon,
            float momentum,
            int axis,
            PairList<String, Object> additional);

    NDList rnn(
            NDList inputs,
            String mode,
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs,
            PairList<String, Object> additional);

    NDList rnn(
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
            PairList<String, Object> additional);

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    /**
     * Picks elements from an input array according to the input indices along the given axis.
     *
     * @param index The index array
     * @param axis The axis to picking the elements. Negative values means indexing from right to
     *     left. If is `None`, the elements in the index w.r.t the flattened input will be picked.
     * @param keepDims If true, the axis where we pick the elements is left in the result as
     *     dimension with size one.
     * @param mode Specify how out-of-bound indices behave. "clip" means clip to the range. So, if
     *     all indices mentioned are too large, they are replaced by the index that addresses the
     *     last element along an axis. "wrap" means to wrap around.
     * @return copy of array
     */
    NDArray pick(NDArray index, int axis, boolean keepDims, String mode);

    /**
     * Picks elements from an input array according to the input indices along the given axis.
     *
     * @param index The index array
     * @param axis The axis to picking the elements. Negative values means indexing from right to
     *     left. If is `None`, the elements in the index w.r.t the flattened input will be picked.
     * @param keepDims If true, the axis where we pick the elements is left in the result as
     *     dimension with size one.
     * @return copy of array
     */
    default NDArray pick(NDArray index, int axis, boolean keepDims) {
        return pick(index, axis, keepDims, "clip");
    }

    NDArray getArray();
}
