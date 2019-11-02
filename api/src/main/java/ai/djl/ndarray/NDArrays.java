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
package ai.djl.ndarray;

import java.util.Arrays;

/** This class contains various methods for manipulating NDArrays. */
public final class NDArrays {

    private NDArrays() {}

    ////////////////////////////////////////
    // Operators: Element Comparison
    ////////////////////////////////////////

    /**
     * Returns the boolean {@code true} iff all elements in the NDArray are equal to the Number.
     *
     * @param a the NDArray to compare
     * @param n the number to compare
     * @return the binary NDArray for "Equals" comparison
     */
    public static boolean equals(NDArray a, Number n) {
        if (a == null) {
            return false;
        }
        return a.contentEquals(n);
    }

    /**
     * Returns the boolean {@code true} iff all elements in both the NDArrays are equal.
     *
     * @param a the NDArray to compare
     * @param b the NDArray to compare
     * @return the binary NDArray for "Equals" comparison
     */
    public static boolean equals(NDArray a, NDArray b) {
        if (a == null) {
            return false;
        }
        return a.contentEquals(b);
    }

    /**
     * Returns the binary NDArray for "Equals" comparison.
     *
     * @param a the NDArray to compare
     * @param n the number to compare
     * @return the binary NDArray for "Equals" comparison
     */
    public static NDArray eq(NDArray a, Number n) {
        return a.eq(n);
    }

    /**
     * Returns the binary NDArray for "Equals" comparison.
     *
     * @param n the number to compare
     * @param a the NDArray to compare
     * @return the binary NDArray for "Equals" comparison
     */
    public static NDArray eq(Number n, NDArray a) {
        return a.eq(n);
    }

    /**
     * Returns the binary NDArray for "Equals" comparison.
     *
     * @param a the NDArray to compare
     * @param b the NDArray to compare
     * @return the binary NDArray for "Equals" comparison
     */
    public static NDArray eq(NDArray a, NDArray b) {
        return a.eq(b);
    }

    /**
     * Returns True if two arrays are element-wise equal within a tolerance.
     *
     * @param a the NDArray to compare with
     * @param b the NDArray to compare with
     * @return the result {@code NDArray}
     */
    public static boolean allClose(NDArray a, NDArray b) {
        return a.allClose(b);
    }

    /**
     * Returns True if two arrays are element-wise equal within a tolerance.
     *
     * @param a the NDArray to compare with
     * @param b the NDArray to compare with
     * @param rtol the relative tolerance parameter
     * @param atol the absolute tolerance parameter
     * @param equalNan whether to compare NaN’s as equal. If True, NaN’s in a will be considered
     *     equal to NaN’s in b in the output array.
     * @return the result {@code NDArray}
     */
    public static boolean allClose(
            NDArray a, NDArray b, double rtol, double atol, boolean equalNan) {
        return a.allClose(b, rtol, atol, equalNan);
    }

    /**
     * Returns the binary NDArray for "Greater Than" comparison.
     *
     * @param a the NDArray to be compared
     * @param n the number to be compared against
     * @return the binary NDArray for "Greater Than" comparison
     */
    public static NDArray gt(NDArray a, Number n) {
        return a.gt(n);
    }

    /**
     * Returns the binary NDArray for "Greater Than" comparison.
     *
     * @param n the number to be compared
     * @param a the NDArray to be compared against
     * @return the binary NDArray for "Greater Than" comparison
     */
    public static NDArray gt(Number n, NDArray a) {
        return a.lt(n);
    }

    /**
     * Returns the binary NDArray for "Greater Than" comparison.
     *
     * @param a the NDArray to be compared
     * @param b the NDArray to be compared against
     * @return the binary NDArray for "Greater Than" comparison
     */
    public static NDArray gt(NDArray a, NDArray b) {
        return a.gt(b);
    }

    /**
     * Returns the binary NDArray for "Greater or equals" comparison.
     *
     * @param a the NDArray to be compared
     * @param n the number to be compared against
     * @return the binary NDArray for "Greater or equals" comparison
     */
    public static NDArray gte(NDArray a, Number n) {
        return a.gte(n);
    }

    /**
     * Returns the binary NDArray for "Greater or equals" comparison.
     *
     * @param n the number to be compared
     * @param a the NDArray to be compared against
     * @return the binary NDArray for "Greater or equals" comparison
     */
    public static NDArray gte(Number n, NDArray a) {
        return a.lte(n);
    }

    /**
     * Returns the binary NDArray for "Greater or Equals" comparison.
     *
     * @param a the NDArray to be compared
     * @param b the NDArray to be compared against
     * @return the binary NDArray for "Greater Than" comparison
     */
    public static NDArray gte(NDArray a, NDArray b) {
        return a.gte(b);
    }

    /**
     * Returns the binary NDArray for "Less" comparison.
     *
     * @param a the NDArray to be compared
     * @param n the number to be compared against
     * @return the binary NDArray for "Less" comparison
     */
    public static NDArray lt(NDArray a, Number n) {
        return a.lt(n);
    }

    /**
     * Returns the binary NDArray for "Less" comparison.
     *
     * @param n the number to be compared
     * @param a the NDArray to be compared against
     * @return the binary NDArray for "Less" comparison
     */
    public static NDArray lt(Number n, NDArray a) {
        return a.gt(n);
    }

    /**
     * Returns the binary NDArray for "Less" comparison.
     *
     * @param a the NDArray to be compared
     * @param b the NDArray to be compared against
     * @return the binary NDArray for "Less" comparison
     */
    public static NDArray lt(NDArray a, NDArray b) {
        return a.lt(b);
    }

    /**
     * Returns the binary NDArray for "Less or equals" comparison.
     *
     * @param a the NDArray to be compared
     * @param n the number to be compared against
     * @return the binary NDArray for "Less or equals" comparison
     */
    public static NDArray lte(NDArray a, Number n) {
        return a.lte(n);
    }

    /**
     * Returns the binary NDArray for "Less or equals" comparison.
     *
     * @param n the number to be compared
     * @param a the NDArray to be compared against
     * @return the binary NDArray for "Less or equals" comparison
     */
    public static NDArray lte(Number n, NDArray a) {
        return a.gte(n);
    }

    /**
     * Returns the binary NDArray for "Lesser or equals" comparison.
     *
     * @param a the NDArray to be compared
     * @param b the NDArray to be compared against
     * @return the binary NDArray for "Less" comparison
     */
    public static NDArray lte(NDArray a, NDArray b) {
        return a.lte(b);
    }

    /**
     * Returns element-wise maximum of the input arrays with broadcasting.
     *
     * @param a the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @param n the number to be compared against
     * @return the maximum of a and b, element-wise
     */
    public static NDArray max(NDArray a, Number n) {
        return a.getNDArrayInternal().max(n);
    }

    /**
     * Returns element-wise maximum of the input arrays with broadcasting.
     *
     * @param n the number to be compared against
     * @param a the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @return the maximum of a and b, element-wise
     */
    public static NDArray max(Number n, NDArray a) {
        return max(a, n);
    }

    /**
     * Returns element-wise maximum of the input arrays with broadcasting.
     *
     * @param a the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @param b the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @return the maximum of a and b, element-wise
     */
    public static NDArray max(NDArray a, NDArray b) {
        return a.getNDArrayInternal().max(b);
    }

    /**
     * Returns element-wise minimum of the input arrays with broadcasting.
     *
     * @param a the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @param n the number to be compared against
     * @return the minimum of a and b, element-wise
     */
    public static NDArray min(NDArray a, Number n) {
        return a.getNDArrayInternal().min(n);
    }

    /**
     * Returns element-wise minimum of the input arrays with broadcasting.
     *
     * @param n the number to be compared against
     * @param a the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @return the minimum of a and b, element-wise
     */
    public static NDArray min(Number n, NDArray a) {
        return min(a, n);
    }

    /**
     * Returns element-wise minimum of the input arrays with broadcasting.
     *
     * @param a the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @param b the NDArray holding the elements to be compared. They must have the same shape, or
     *     shapes that can be broadcast to a single shape.
     * @return the minimum of a and b, element-wise
     */
    public static NDArray min(NDArray a, NDArray b) {
        return a.getNDArrayInternal().min(b);
    }

    /**
     * Returns the elements, either from this NDArray or other, depending on the condition.
     *
     * @param condition the condition array
     * @param a the first NDArray
     * @param b the other NDArray
     * @return the result NDArray
     */
    public static NDArray where(NDArray condition, NDArray a, NDArray b) {
        return a.where(condition, b);
    }

    /**
     * Returns partial of {@code NDArray} based on boolean index {@code NDArray} from axis 0.
     *
     * @param data the data to operate on
     * @param index the boolean {@code NDArray} mask
     * @return the new {@code NDArray}
     */
    public static NDArray booleanMask(NDArray data, NDArray index) {
        return booleanMask(data, index, 0);
    }

    /**
     * Returns partial of {@code NDArray} based on boolean index {@code NDArray} and axis.
     *
     * @param data the data to operate on
     * @param index the boolean {@code NDArray} mask
     * @param axis an integer that represents the axis in {@code NDArray} to mask from
     * @return the new {@code NDArray}
     */
    public static NDArray booleanMask(NDArray data, NDArray index, int axis) {
        return data.booleanMask(index, axis);
    }

    ////////////////////////////////////////
    // Operators: Element Arithmetic
    ////////////////////////////////////////

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to
     * @param n the number to add to the {@link NDArray} elements
     * @return the result of the addition
     */
    public static NDArray add(NDArray a, Number n) {
        return a.add(n);
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to
     * @param n the number to add to the {@link NDArray} elements
     * @return the result of the addition
     */
    public static NDArray add(Number n, NDArray a) {
        return a.add(n);
    }

    /**
     * Adds {@link NDArray}s element-wise with broadcasting.
     *
     * @param arrays the arrays to add together
     * @return the result of the addition
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static NDArray add(NDArray... arrays) {
        if (arrays == null || arrays.length < 2) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        return arrays[0].add(Arrays.stream(arrays).skip(1).toArray(NDArray[]::new));
    }

    /**
     * Performs scalar subtraction of an NDArray (copied).
     *
     * @param a the NDArray to be operated on
     * @param n the number to subtract by
     * @return a copy of this array after applying subtraction operation
     */
    public static NDArray sub(NDArray a, Number n) {
        return a.sub(n);
    }

    /**
     * Subtracts an NDArray with duplicates - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for subtraction
     * @param a the NDArray to be operated on
     * @return a copy of the array after subtraction
     */
    public static NDArray sub(Number n, NDArray a) {
        return a.getNDArrayInternal().rsub(n);
    }

    /**
     * Performs scalar subtraction (copied).
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray to subtract by
     * @return a copy of this array after applying subtraction operation
     */
    public static NDArray sub(NDArray a, NDArray b) {
        return a.sub(b);
    }

    /**
     * Performs scalar multiplication (copy).
     *
     * @param a the NDArray to be operated on
     * @param n the number to multiply by
     * @return a copy of this NDArray multiplied by the given number
     */
    public static NDArray mul(NDArray a, Number n) {
        return a.mul(n);
    }

    /**
     * Performs scalar multiplication (copy).
     *
     * @param n the number to multiply by
     * @param a the NDArray to be operated on
     * @return a copy of this NDArray multiplied by the given number
     */
    public static NDArray mul(Number n, NDArray a) {
        return a.mul(n);
    }

    /**
     * Multiplies {@link NDArray}s element-wise with broadcasting.
     *
     * @param arrays the arrays to multiply together
     * @return the result of the multiplication
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static NDArray mul(NDArray... arrays) {
        if (arrays == null || arrays.length < 2) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        return arrays[0].mul(Arrays.stream(arrays).skip(1).toArray(NDArray[]::new));
    }

    /**
     * Divides an NDArray by a number.
     *
     * @param a the NDArray to be operated on
     * @param n the number to divide values by
     * @return a copy of the array after division
     */
    public static NDArray div(NDArray a, Number n) {
        return a.div(n);
    }

    /**
     * Divides an NDArray with a scalar - i.e., (n / thisArrayValues).
     *
     * @param n the value to use for division
     * @param a the NDArray to be operated on
     * @return a copy of the array after applying division
     */
    public static NDArray div(Number n, NDArray a) {
        return a.getNDArrayInternal().rdiv(n);
    }

    /**
     * Performs in-place scalar division.
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray to divide values by
     * @return this array after applying division operation
     */
    public static NDArray div(NDArray a, NDArray b) {
        return a.div(b);
    }

    /**
     * Returns the scalar remainder of division (copy).
     *
     * @param a the NDArray to be operated on
     * @param n the number to multiply by
     * @return a copy of this NDArray multiplied by the given number
     */
    public static NDArray mod(NDArray a, Number n) {
        return a.mod(n);
    }

    /**
     * Copies the scalar remainder of division.
     *
     * @param n the number to multiply by
     * @param a the NDArray to be operated on
     * @return a copy of this NDArray multiplied by the given number
     */
    public static NDArray mod(Number n, NDArray a) {
        return a.getNDArrayInternal().rmod(n);
    }

    /**
     * Copies the scalar remainder of division.
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray mod(NDArray a, NDArray b) {
        return a.mod(b);
    }

    /**
     * Raises the power of each element in the NDArray.
     *
     * @param a the NDArray to be operated on
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    public static NDArray pow(NDArray a, Number n) {
        return a.pow(n);
    }

    /**
     * Raises the power of each element in the NDArray by the corresponding element in the other
     * NDArray.
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray by which the raise the power by
     * @return the result {@code NDArray}
     */
    public static NDArray pow(NDArray a, NDArray b) {
        return a.pow(b);
    }

    /**
     * Raises the power of a number by the corresponding element in the {@code NDArray}.
     *
     * @param n the number to be operated on
     * @param a the {@code NDArray} by which the raise the power by
     * @return the result {@code NDArray}
     */
    public static NDArray pow(Number n, NDArray a) {
        return a.getNDArrayInternal().rpow(n);
    }

    /**
     * Adds a number to each element of an {@link NDArray} in-place.
     *
     * @param a the NDArray that will be added to
     * @param n the number to add to the {@link NDArray} elements
     * @return the result of the addition
     */
    public static NDArray addi(NDArray a, Number n) {
        return a.addi(n);
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to
     * @param n the number to add to the {@link NDArray} elements
     * @return the result of the addition
     */
    public static NDArray addi(Number n, NDArray a) {
        return a.addi(n);
    }

    /**
     * Adds {@link NDArray}s element-wise with broadcasting.
     *
     * @param arrays the arrays to add together
     * @return the result of the addition
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static NDArray addi(NDArray... arrays) {
        if (arrays == null || arrays.length < 2) {
            throw new IllegalArgumentException("Passed in arrays must have at least two elements");
        }
        return arrays[0].addi(Arrays.stream(arrays).skip(1).toArray(NDArray[]::new));
    }

    /**
     * Performs in-place scalar subtraction of an NDArray.
     *
     * @param a the NDArray to be operated on
     * @param n the number to subtract
     * @return this array after applying subtraction operation
     */
    public static NDArray subi(NDArray a, Number n) {
        return a.subi(n);
    }

    /**
     * Subtracts an NDArray in place - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for subtraction
     * @param a the NDArray to be operated on
     * @return this array after subtraction
     */
    public static NDArray subi(Number n, NDArray a) {
        return a.getNDArrayInternal().rsubi(n);
    }

    /**
     * Performs in-place scalar subtraction.
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray to subtract by
     * @return this array after applying subtraction operation
     */
    public static NDArray subi(NDArray a, NDArray b) {
        return a.subi(b);
    }

    /**
     * Performs in-place scalar multiplication.
     *
     * @param a the NDArray to be operated on
     * @param n the number to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray muli(NDArray a, Number n) {
        return a.muli(n);
    }

    /**
     * Performs in-place scalar multiplication.
     *
     * @param n the number to multiply by
     * @param a the NDArray to be operated on
     * @return this array after applying scalar multiplication
     */
    public static NDArray muli(Number n, NDArray a) {
        return a.muli(n);
    }

    /**
     * Multiplies {@link NDArray}s in place element-wise with broadcasting.
     *
     * @param arrays the arrays to multiply together
     * @return the result of the multiplication
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static NDArray muli(NDArray... arrays) {
        if (arrays == null || arrays.length < 2) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        return arrays[0].muli(Arrays.stream(arrays).skip(1).toArray(NDArray[]::new));
    }

    /**
     * Divides an NDArray by a number.
     *
     * @param a the NDArray to be operated on
     * @param n the number to divide values by
     * @return a copy of the array after division
     */
    public static NDArray divi(NDArray a, Number n) {
        return a.divi(n);
    }

    /**
     * Divides an NDArray - i.e., (n / thisArrayValues) in-place.
     *
     * @param n the value to use for division
     * @param a the NDArray to be operated on
     * @return this array after applying division
     */
    public static NDArray divi(Number n, NDArray a) {
        return a.getNDArrayInternal().rdivi(n);
    }

    /**
     * Performs in-place scalar division.
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray to divide values by
     * @return this array after applying division operation
     */
    public static NDArray divi(NDArray a, NDArray b) {
        return a.divi(b);
    }

    /**
     * Returns the in-place remainder of division.
     *
     * @param a the NDArray to be operated on
     * @param n the number to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray modi(NDArray a, Number n) {
        return a.modi(n);
    }

    /**
     * Returns the in-place scalar remainder of division.
     *
     * @param n the number to multiply by
     * @param a the NDArray to be operated on
     * @return this array after applying scalar multiplication
     */
    public static NDArray modi(Number n, NDArray a) {
        return a.getNDArrayInternal().rmodi(n);
    }

    /**
     * Returns the in-place scalar remainder of division.
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray to multiply by
     * @return this array after applying scalar multiplication
     */
    public static NDArray modi(NDArray a, NDArray b) {
        return a.modi(b);
    }

    /**
     * Raises the power of each element in the NDArray in-place.
     *
     * @param a the NDArray to be operated on
     * @param n the number to raise the power to
     * @return this {@code NDArray} after applying raising power
     */
    public static NDArray powi(NDArray a, Number n) {
        return a.powi(n);
    }

    /**
     * Raises the power of a number by the corresponding element in the {@code NDArray}.
     *
     * @param n the number to be operated on
     * @param a the {@code NDArray} by which the raise the power by
     * @return this {@code NDArray} after applying raising power
     */
    public static NDArray powi(Number n, NDArray a) {
        return a.getNDArrayInternal().rpowi(n);
    }

    /**
     * Raises the power of each element in the NDArray by the corresponding element in the other
     * NDArray in-place.
     *
     * @param a the NDArray to be operated on
     * @param b the NDArray to raise the power by
     * @return the result {@code NDArray}
     */
    public static NDArray powi(NDArray a, NDArray b) {
        return a.powi(b);
    }

    /**
     * Copies matrix multiplication of two NDArrays.
     *
     * @param a the NDArray to be operated on
     * @param b the second NDArray to multiply
     * @return the result of the addition
     */
    public static NDArray dot(NDArray a, NDArray b) {
        return a.dot(b);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last
     * dimension.
     *
     * @param arrays the input {@link NDList}. Each {@link NDArray} must have the same shape.
     * @param axis the axis in the result array along which the input arrays are stacked
     * @return the {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDList arrays, int axis) {
        NDArray array = arrays.head();
        return array.stack(arrays.subNDList(1), axis);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along axis 0.
     *
     * @param arrays the input NDList. Each {@link NDArray} must have the same shape.
     * @return the {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along existing axis.
     *
     * @param arrays the input NDList. Each {@link NDArray} must have the same shape.
     * @param axis the axis along which the arrays will be joined
     * @return the {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDList arrays, int axis) {
        NDArray array = arrays.head();
        return array.concat(arrays.subNDList(1), axis);
    }

    /**
     * Joins a sequence of {@link NDArray} in NDList along axis 0.
     *
     * @param arrays the input NDList. Each {@link NDArray} must have the same shape.
     * @return the {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDList arrays) {
        return concat(arrays, 0);
    }

    /**
     * Computes the truth value of {@code NDArray} a AND {@code NDArray} b element-wise.
     *
     * @param a the input {@code NDArray}
     * @param b the other input {@code NDArray}
     * @return the boolean result of the logical AND operation applied to the elements of two {@code
     *     NDArray}
     */
    public static NDArray logicalAnd(NDArray a, NDArray b) {
        return a.logicalAnd(b);
    }

    /**
     * Computes the truth value of {@code NDArray} a OR {@code NDArray} b element-wise.
     *
     * @param a the input {@code NDArray}
     * @param b the other input {@code NDArray}
     * @return the boolean result of the logical OR operation applied to the elements of two {@code
     *     NDArray}
     */
    public static NDArray logicalOr(NDArray a, NDArray b) {
        return a.logicalOr(b);
    }

    /**
     * Computes the truth value of {@code NDArray} a XOR {@code NDArray} b element-wise.
     *
     * @param a the input {@code NDArray}
     * @param b the other input {@code NDArray}
     * @return the boolean result of the logical XOR operation applied to the elements of two {@code
     *     NDArray}
     */
    public static NDArray logicalXor(NDArray a, NDArray b) {
        return a.logicalXor(b);
    }
}
