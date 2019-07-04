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
package software.amazon.ai.ndarray;

import java.util.Arrays;

/** This class contains various methods for manipulating NDArrays. */
public final class NDArrays {

    private NDArrays() {}

    /**
     * Returns the binary ndarray for "Equals" comparison.
     *
     * @param a the ndarray to compare.
     * @param n the number to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    public static NDArray eq(NDArray a, Number n) {
        return a.eq(n);
    }

    /**
     * Returns the binary ndarray for "Equals" comparison.
     *
     * @param a the ndarray to compare.
     * @param b the ndarray to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    public static NDArray eq(NDArray a, NDArray b) {
        return a.eq(b);
    }

    /**
     * Returns the boolean {@code true} iff all elements in both the NDArrays are equal.
     *
     * @param a the ndarray to compare.
     * @param b the ndarray to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    public static boolean equals(NDArray a, NDArray b) {
        return a.contentEquals(b);
    }

    /**
     * Returns the boolean {@code true} iff all elements in the NDArray is equal to the Number.
     *
     * @param a the ndarray to compare.
     * @param b the number to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    public static boolean equals(NDArray a, Number b) {
        return a.contentEquals(b);
    }

    /**
     * Returns the binary ndarray for "Greater Than" comparison.
     *
     * @param a ndarray to be compared
     * @param b the ndarray to be compared against
     * @return the binary ndarray for "Greater Than" comparison.
     */
    public static NDArray gt(NDArray a, NDArray b) {
        return a.gt(b);
    }

    /**
     * Returns the binary ndarray for "Greater Than" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return binary ndarray for "Greater Than" comparison.
     */
    public static NDArray gt(NDArray a, Number b) {
        return a.gt(b);
    }

    /**
     * Returns the binary ndarray for "Greater or Equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the ndarray to be compared against
     * @return the binary ndarray for "Greater Than" comparison.
     */
    public static NDArray gte(NDArray a, NDArray b) {
        return a.gte(b);
    }

    /**
     * Returns the binary ndarray for "Greater or equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return binary ndarray for "Greater or equals" comparison.
     */
    public static NDArray gte(NDArray a, Number b) {
        return a.gte(b);
    }

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return the binary ndarray for "Less" comparison.
     */
    public static NDArray lt(NDArray a, Number b) {
        return a.lt(b);
    }

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param a ndarray to be compared
     * @param b the ndarray to be compared against
     * @return the binary ndarray for "Less" comparison.
     */
    public static NDArray lt(NDArray a, NDArray b) {
        return a.lt(b);
    }

    /**
     * Returns the binary ndarray for "Lesser or equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the ndarray to be compared against
     * @return the binary ndarray for "Less" comparison.
     */
    public static NDArray lte(NDArray a, NDArray b) {
        return a.lte(b);
    }

    /**
     * Returns the binary ndarray for "Less or equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return the binary ndarray for "Less or equals" comparison.
     */
    public static NDArray lte(NDArray a, Number b) {
        return a.lte(b);
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return the result of the addition
     */
    public static NDArray add(NDArray a, Number n) {
        return a.add(n);
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return the result of the addition
     */
    public static NDArray add(Number n, NDArray a) {
        return a.add(n);
    }

    /**
     * Adds two {@link NDArray}s with broadcasting.
     *
     * @param a the left NDArray
     * @param b the right NDArray
     * @return the result of the addition
     */
    public static NDArray add(NDArray a, NDArray b) {
        return a.add(b);
    }

    /**
     * In place Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return the result of the addition
     */
    public static NDArray addi(NDArray a, Number n) {
        return a.addi(n);
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return the result of the addition
     */
    public static NDArray addi(Number n, NDArray a) {
        return a.addi(n);
    }

    /**
     * Adds two {@link NDArray}s with broadcasting.
     *
     * @param a the left NDArray
     * @param b the right NDArray
     * @return the result of the addition
     */
    public static NDArray addi(NDArray a, NDArray b) {
        return a.addi(b);
    }

    /**
     * Divides an NDArray by a number.
     *
     * @param a ndarray to be operated on
     * @param n Number to divide values by
     * @return Copy of array after division
     */
    public static NDArray div(NDArray a, Number n) {
        return a.div(n);
    }

    /**
     * Divides an NDArray with a scalar - i.e., (n / thisArrayValues).
     *
     * @param n Value to use for division
     * @param a ndarray to be operated on
     * @return Copy of array after applying division
     */
    public static NDArray div(Number n, NDArray a) {
        return a.getNDArrayInternal().rdiv(n);
    }

    /**
     * In place scalar division.
     *
     * @param a ndarray to be operated on
     * @param b ndarray to divide values by
     * @return this array after applying division operation
     */
    public static NDArray div(NDArray a, NDArray b) {
        return a.div(b);
    }

    /**
     * Divides an NDArray by a number.
     *
     * @param a ndarray to be operated on
     * @param n Number to divide values by
     * @return Copy of array after division
     */
    public static NDArray divi(NDArray a, Number n) {
        return a.divi(n);
    }

    /**
     * In place divides an NDArray - i.e., (n / thisArrayValues).
     *
     * @param n Value to use for division
     * @param a ndarray to be operated on
     * @return This array after applying division
     */
    public static NDArray divi(Number n, NDArray a) {
        return a.getNDArrayInternal().rdivi(n);
    }

    /**
     * In place scalar division.
     *
     * @param a ndarray to be operated on
     * @param b ndarray to divide values by
     * @return this array after applying division operation
     */
    public static NDArray divi(NDArray a, NDArray b) {
        return a.divi(b);
    }

    /**
     * Scalar subtraction of an NDArray (copied).
     *
     * @param a ndarray to be operated on
     * @param n the number to subtract by
     * @return Copy of this array after applying subtraction operation
     */
    public static NDArray sub(NDArray a, Number n) {
        return a.sub(n);
    }

    /**
     * Subtracts an NDArray with duplicates - i.e., (n - thisArrayValues).
     *
     * @param n Value to use for subtraction
     * @param a ndarray to be operated on
     * @return Copy of array after subtraction
     */
    public static NDArray sub(Number n, NDArray a) {
        return a.getNDArrayInternal().rsub(n);
    }

    /**
     * Scalar subtraction (copied).
     *
     * @param a ndarray to be operated on
     * @param b the ndarray to subtract by
     * @return Copy of this array after applying subtraction operation
     */
    public static NDArray sub(NDArray a, NDArray b) {
        return a.sub(b);
    }

    /**
     * In place scalar subtraction of an NDArray.
     *
     * @param a ndarray to be operated on
     * @param n Number to subtract
     * @return this array after applying subtraction operation
     */
    public static NDArray subi(NDArray a, Number n) {
        return a.subi(n);
    }

    /**
     * Subtracts an NDArray in place - i.e., (n - thisArrayValues).
     *
     * @param n Value to use for subtraction
     * @param a ndarray to be operated on
     * @return This array after subtraction
     */
    public static NDArray subi(Number n, NDArray a) {
        return a.getNDArrayInternal().rsubi(n);
    }

    /**
     * In place scalar subtraction.
     *
     * @param a ndarray to be operated on
     * @param b the ndarray to subtract by
     * @return this array after applying subtraction operation
     */
    public static NDArray subi(NDArray a, NDArray b) {
        return a.subi(b);
    }

    /**
     * Scalar multiplication (copy).
     *
     * @param a ndarray to be operated on
     * @param n the number to multiply by
     * @return a copy of this ndarray multiplied by the given number
     */
    public static NDArray mul(NDArray a, Number n) {
        return a.mul(n);
    }

    /**
     * Scalar multiplication (copy).
     *
     * @param n the number to multiply by
     * @param a ndarray to be operated on
     * @return a copy of this ndarray multiplied by the given number
     */
    public static NDArray mul(Number n, NDArray a) {
        return a.mul(n);
    }

    /**
     * Copies (element wise) multiplication of two NDArrays.
     *
     * @param a ndarray to be operated on
     * @param b the second NDArray to multiply
     * @return the result of the addition
     */
    public static NDArray mul(NDArray a, NDArray b) {
        return a.mul(b);
    }

    /**
     * In place scalar multiplication.
     *
     * @param a ndarray to be operated on
     * @param n The number to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray muli(NDArray a, Number n) {
        return a.muli(n);
    }

    /**
     * In place scalar multiplication.
     *
     * @param n The number to multiply by
     * @param a ndarray to be operated on
     * @return this array after applying scalar multiplication
     */
    public static NDArray muli(Number n, NDArray a) {
        return a.muli(n);
    }

    /**
     * In place scalar multiplication.
     *
     * @param a ndarray to be operated on
     * @param b ndarray to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray muli(NDArray a, NDArray b) {
        return a.muli(b);
    }

    /**
     * HEAD Raises the power of each element in the ndarray.
     *
     * @param a ndarray to be operated on
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    public static NDArray pow(NDArray a, Number n) {
        return a.pow(n);
    }

    /**
     * Raises the power of each element in the ndarray by the corresponding element in the other
     * ndarray.
     *
     * @param a ndarray to be operated on
     * @param b the ndarray by which the raise the power by
     * @return the result {@code NDArray}
     */
    public static NDArray pow(NDArray a, NDArray b) {
        return a.pow(b);
    }

    /**
     * Raises the power of each element in the ndarray in-place.
     *
     * @param a ndarray to be operated on
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    public static NDArray powi(NDArray a, Number n) {
        return a.powi(n);
    }

    /**
     * Raises the power of each element in the ndarray by the corresponding element in the other
     * ndarray in-place.
     *
     * @param a ndarray to be operated on
     * @param b the ndarray by which the raise the power by
     * @return the result {@code NDArray}
     */
    public static NDArray powi(NDArray a, NDArray b) {
        return a.powi(b);
    }

    /**
     * Scalar remainder of division (copy).
     *
     * @param a ndarray to be operated on
     * @param n the number to multiply by
     * @return a copy of this ndarray multiplied by the given number
     */
    public static NDArray mod(NDArray a, Number n) {
        return a.mod(n);
    }

    /**
     * Copy scalar remainder of division.
     *
     * @param n the number to multiply by
     * @param a ndarray to be operated on
     * @return a copy of this ndarray multiplied by the given number
     */
    public static NDArray mod(Number n, NDArray a) {
        return a.getNDArrayInternal().rmod(n);
    }

    /**
     * Scalar remainder of division.
     *
     * @param a ndarray to be operated on
     * @param b ndarray to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray mod(NDArray a, NDArray b) {
        return a.mod(b);
    }

    /**
     * In place remainder of division.
     *
     * @param a ndarray to be operated on
     * @param n the number to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray modi(NDArray a, Number n) {
        return a.modi(n);
    }

    /**
     * In place scalar remainder of division.
     *
     * @param n The number to multiply by
     * @param a ndarray to be operated on
     * @return this array after applying scalar multiplication
     */
    public static NDArray modi(Number n, NDArray a) {
        return a.getNDArrayInternal().rmodi(n);
    }

    /**
     * In place scalar remainder of division.
     *
     * @param a ndarray to be operated on
     * @param b ndarray to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray modi(NDArray a, NDArray b) {
        return a.modi(b);
    }

    /**
     * copy matrix multiplication of two NDArrays.
     *
     * @param a ndarray to be operated on
     * @param b the second NDArray to multiply
     * @return the result of the addition
     */
    public static NDArray mmul(NDArray a, NDArray b) {
        return a.mmul(b);
    }

    /**
     * Join a sequence of {@link NDArray} along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last
     * dimension.
     *
     * @param arrays input {@link NDArray}[]. each {@link NDArray} must have the same shape.
     * @param axis the axis in the result array along which the input arrays are stacked.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDArray[] arrays, int axis) {
        NDArray array = arrays[0];
        return array.stack(Arrays.copyOfRange(arrays, 1, arrays.length), axis);
    }

    /**
     * Joins a sequence of {@link NDArray} along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDArray[] arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last
     * dimension.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @param axis the axis in the result array along which the input arrays are stacked.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDList arrays, int axis) {
        return stack(arrays.toArray(), axis);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDList arrays) {
        return stack(arrays.toArray(), 0);
    }

    /**
     * Joins a sequence of {@link NDArray} along an existing axis.
     *
     * @param arrays the arrays must have the same shape, except in the dimension corresponding to
     *     `axis` (the first, by default).
     * @param axis the axis along which the arrays will be joined.
     * @return the concatenated array
     */
    public static NDArray concat(NDArray[] arrays, int axis) {
        NDArray array = arrays[0];
        return array.concat(Arrays.copyOfRange(arrays, 1, arrays.length), axis);
    }

    /**
     * Joins a sequence of {@link NDArray} along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDArray[] arrays) {
        return concat(arrays, 0);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along existing axis.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @param axis the axis along which the arrays will be joined.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDList arrays, int axis) {
        return concat(arrays.toArray(), axis);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDList arrays) {
        return concat(arrays.toArray(), 0);
    }
}
