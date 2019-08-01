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

import java.util.stream.IntStream;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.pooling.PoolingConvention;

/** An internal interface that encapsulate engine specific operator methods. */
public interface NDArrayEx {

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

    /**
     * Computes rectified linear activation.
     *
     * @return copy of array after applying relu
     */
    NDArray relu();

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
     * @param other the arrays holding the elements to be compared. They must have the same shape,
     *     or shapes that can be broadcast to a single shape.
     * @return the maximum of two {@code NDArray}.
     */
    NDArray max(NDArray other);

    /**
     * Returns element-wise minimum of the input arrays with broadcasting.
     *
     * @param other the arrays holding the elements to be compared. They must have the same shape,
     *     or shapes that can be broadcast to a single shape.
     * @return the minimum of two {@code NDArray}.
     */
    NDArray min(NDArray other);

    NDArray maxPool(Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention);

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

    void sgdUpdate(
            NDArray grad,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGradient,
            boolean lazyUpdate);

    void sgdMomUpdate(
            NDArray grad,
            NDArray state,
            float lr,
            float wd,
            float momentum,
            float rescaleGrad,
            float clipGradient,
            boolean lazyUpdate);

    NDArray getArray();

    default NDArray l2Loss(NDArray label, float weight, int batchAxis) {
        NDArray pred = getArray();
        label = label.reshape(pred.getShape());
        NDArray loss = label.sub(pred).square().mul(weight);
        return loss.mean(new int[] {batchAxis});
    }

    default NDArray softmaxCrossEntropyLoss(
            NDArray label,
            float weight,
            int batchAxis,
            int classAxis,
            boolean sparseLabel,
            boolean fromLogit) {
        NDArray pred = getArray();
        if (!fromLogit) {
            pred = pred.softmax(classAxis).log();
        }
        NDArray loss;
        if (sparseLabel) {
            loss = pred.getNDArrayInternal().pick(label, classAxis, true).neg();
        } else {
            label = label.reshape(pred.getShape());
            loss = pred.mul(label).sum(new int[] {classAxis}).mul(-weight);
        }
        // apply mean on all axes except the batchAxis
        int[] axes =
                IntStream.range(0, loss.getShape().dimension())
                        .filter(axis -> axis != batchAxis)
                        .toArray();
        return loss.mean(axes);
    }
}
