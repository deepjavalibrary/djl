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

/** An internal interface that encapsulate engine specific operator methods. */
public interface NDArrayEx {

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
            int axis,
            boolean sparseLabel,
            boolean fromLogit) {
        NDArray pred = getArray();
        if (!fromLogit) {
            pred = pred.softmax(axis).log();
        }
        if (!sparseLabel) {
            label = label.toDense();
        }
        label = label.reshape(pred.getShape());
        NDArray loss = pred.mmul(label).sum(new int[] {axis}).mul(-weight);
        return loss.mean(new int[] {batchAxis});
    }
}
