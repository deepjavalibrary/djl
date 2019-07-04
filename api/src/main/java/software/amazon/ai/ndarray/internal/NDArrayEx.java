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
}
