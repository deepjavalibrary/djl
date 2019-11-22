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

import ai.djl.ndarray.types.Shape;
import java.util.Arrays;

/** This class contains various methods for manipulating NDArrays. */
public final class NDArrays {

    private NDArrays() {}

    private static void checkInputs(NDArray[] arrays) {
        if (arrays == null || arrays.length < 2) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        if (arrays.length > 2
                && Arrays.stream(arrays).skip(1).anyMatch(array -> !arrays[0].shapeEquals(array))) {
            throw new IllegalArgumentException("The shape of all inputs must be the same");
        }
    }

    ////////////////////////////////////////
    // Operators: Element Comparison
    ////////////////////////////////////////

    /**
     * Returns {@code true} if all elements in {@link NDArray} a are equal to {@link NDArray} b.
     *
     * @param a the {@link NDArray} to compare
     * @param b the {@link NDArray} to compare
     * @return the boolean result
     */
    public static boolean contenEquals(NDArray a, NDArray b) {
        return a.contentEquals(b);
    }

    /**
     * Checks 2 {@link NDArray}s for equal shapes.
     *
     * <p>Shapes are considered equal if:
     *
     * <ul>
     *   <li>Both {@link NDArray}s have equal rank, and
     *   <li>size(0)...size(rank()-1) are equal for both {@link NDArray}s
     * </ul>
     *
     * @param a the {@link NDArray} to compare
     * @param b the {@link NDArray} to compare
     * @return {@code true} if the {@link Shape}s are the same
     */
    public static boolean shapeEquals(NDArray a, NDArray b) {
        return a.shapeEquals(b);
    }

    /**
     * Returns {@code true} if all elements in the {@link NDArray} are equal to the Number.
     *
     * @param a the {@link NDArray} to compare
     * @param n the number to compare
     * @return the result boolean
     */
    public static boolean equals(NDArray a, Number n) {
        if (a == null) {
            return false;
        }
        return a.contentEquals(n);
    }

    /**
     * Returns {@code true} if all elements in both the {@link NDArray}s are equal.
     *
     * @param a the {@link NDArray} to compare
     * @param b the {@link NDArray} to compare
     * @return the result boolean
     */
    public static boolean equals(NDArray a, NDArray b) {
        if (a == null) {
            return false;
        }
        return a.contentEquals(b);
    }

    /**
     * Returns {@code true} if two {@link NDArray} are element-wise equal within a tolerance.
     *
     * @param a the {@link NDArray} to compare with
     * @param b the {@link NDArray} to compare with
     * @return the boolean result
     */
    public static boolean allClose(NDArray a, NDArray b) {
        return a.allClose(b);
    }

    /**
     * Returns {@code true} if two {@link NDArray} are element-wise equal within a tolerance.
     *
     * @param a the {@link NDArray} to compare with
     * @param b the {@link NDArray} to compare with
     * @param rtol the relative tolerance parameter
     * @param atol the absolute tolerance parameter
     * @param equalNan whether to compare NaN’s as equal. If {@code true}, NaN’s in the {@link
     *     NDArray} will be considered equal to NaN’s in the other {@link NDArray}
     * @return the boolean result
     */
    public static boolean allClose(
            NDArray a, NDArray b, double rtol, double atol, boolean equalNan) {
        return a.allClose(b, rtol, atol, equalNan);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Equals" comparison.
     *
     * @param a the {@link NDArray} to compare
     * @param n the number to compare
     * @return the boolean {@link NDArray} for element-wise "Equals" comparison
     */
    public static NDArray eq(NDArray a, Number n) {
        return a.eq(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Equals" comparison.
     *
     * @param n the number to compare
     * @param a the {@link NDArray} to compare
     * @return the boolean {@link NDArray} for element-wise "Equals" comparison
     */
    public static NDArray eq(Number n, NDArray a) {
        return a.eq(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Equals" comparison.
     *
     * @param a the {@link NDArray} to compare
     * @param b the {@link NDArray} to compare
     * @return the boolean {@link NDArray} for element-wise "Equals" comparison
     */
    public static NDArray eq(NDArray a, NDArray b) {
        return a.eq(b);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Not equals" comparison.
     *
     * @param a the {@link NDArray} to compare
     * @param n the number to compare
     * @return the boolean {@link NDArray} for element-wise "Not equals" comparison
     */
    public static NDArray neq(NDArray a, Number n) {
        return a.neq(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Not equals" comparison.
     *
     * @param n the number to compare
     * @param a the {@link NDArray} to compare
     * @return the boolean {@link NDArray} for element-wise "Not equals" comparison
     */
    public static NDArray neq(Number n, NDArray a) {
        return a.neq(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Not equals" comparison.
     *
     * @param a the {@link NDArray} to compare
     * @param b the {@link NDArray} to compare
     * @return the boolean {@link NDArray} for element-wise "Not equals" comparison
     */
    public static NDArray neq(NDArray a, NDArray b) {
        return a.neq(b);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Greater Than" comparison.
     *
     * @param a the {@link NDArray} to compare
     * @param n the number to be compared against
     * @return the boolean {@link NDArray} for element-wise "Greater Than" comparison
     */
    public static NDArray gt(NDArray a, Number n) {
        return a.gt(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Greater Than" comparison.
     *
     * @param n the number to be compared
     * @param a the NDArray to be compared against
     * @return the boolean {@link NDArray} for element-wise "Greater Than" comparison
     */
    public static NDArray gt(Number n, NDArray a) {
        return a.lt(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Greater Than" comparison.
     *
     * @param a the {@link NDArray} to be compared
     * @param b the {@link NDArray} to be compared against
     * @return the boolean {@link NDArray} for element-wise "Greater Than" comparison
     */
    public static NDArray gt(NDArray a, NDArray b) {
        return a.gt(b);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Greater or equals" comparison.
     *
     * @param a the {@link NDArray} to be compared
     * @param n the number to be compared against
     * @return the boolean {@link NDArray} for element-wise "Greater or equals" comparison
     */
    public static NDArray gte(NDArray a, Number n) {
        return a.gte(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Greater or equals" comparison.
     *
     * @param n the number to be compared
     * @param a the {@link NDArray} to be compared against
     * @return the boolean {@link NDArray} for element-wise "Greater or equals" comparison
     */
    public static NDArray gte(Number n, NDArray a) {
        return a.lte(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Greater or equals" comparison.
     *
     * @param a the {@link NDArray} to be compared
     * @param b the {@link NDArray} to be compared against
     * @return the boolean {@link NDArray} for element-wise "Greater or equals" comparison
     */
    public static NDArray gte(NDArray a, NDArray b) {
        return a.gte(b);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Less" comparison.
     *
     * @param a the {@link NDArray} to be compared
     * @param n the number to be compared against
     * @return the boolean {@link NDArray} for element-wise "Less" comparison
     */
    public static NDArray lt(NDArray a, Number n) {
        return a.lt(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Less" comparison.
     *
     * @param n the number to be compared
     * @param a the {@link NDArray} to be compared against
     * @return the boolean {@link NDArray} for element-wise "Less" comparison
     */
    public static NDArray lt(Number n, NDArray a) {
        return a.gt(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Less" comparison.
     *
     * @param a the {@link NDArray} to be compared
     * @param b the {@link NDArray} to be compared against
     * @return the boolean {@link NDArray} for element-wise "Less" comparison
     */
    public static NDArray lt(NDArray a, NDArray b) {
        return a.lt(b);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Less or equals" comparison.
     *
     * @param a the {@link NDArray} to be compared
     * @param n the number to be compared against
     * @return the boolean {@link NDArray} for element-wise "Less or equals" comparison
     */
    public static NDArray lte(NDArray a, Number n) {
        return a.lte(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Less or equals" comparison.
     *
     * @param n the number to be compared
     * @param a the {@link NDArray} to be compared against
     * @return the boolean {@link NDArray} for element-wise "Less or equals" comparison
     */
    public static NDArray lte(Number n, NDArray a) {
        return a.gte(n);
    }

    /**
     * Returns the boolean {@link NDArray} for element-wise "Less or equals" comparison.
     *
     * @param a the {@link NDArray} to be compared
     * @param b the {@link NDArray} to be compared against
     * @return the boolean {@link NDArray} for element-wise "Less or equals" comparison
     */
    public static NDArray lte(NDArray a, NDArray b) {
        return a.lte(b);
    }

    /**
     * Returns elements chosen from the {@link NDArray} or the other {@link NDArray} depending on
     * condition.
     *
     * <p>Given three {@link NDArray}s, condition, a, and b, returns an {@link NDArray} with the
     * elements from a or b, depending on whether the elements from condition {@link NDArray} are
     * {@code true} or {@code false}. If condition has the same shape as a, each element in the
     * output {@link NDArray} is from this if the corresponding element in the condition is {@code
     * true}, and from other if {@code false}.
     *
     * <p>Note that all non-zero values are interpreted as {@code true} in condition {@link
     * NDArray}.
     *
     * @param condition the condition {@code NDArray}
     * @param a the first {@link NDArray}
     * @param b the other {@link NDArray}
     * @return the result {@link NDArray}
     */
    public static NDArray where(NDArray condition, NDArray a, NDArray b) {
        return a.getNDArrayInternal().where(condition, b);
    }

    /**
     * Returns the maximum of a {@link NDArray} and a number element-wise.
     *
     * @param a the {@link NDArray} to be compared
     * @param n the number to be compared
     * @return the maximum of a {@link NDArray} and a number element-wise
     */
    public static NDArray maximum(NDArray a, Number n) {
        return a.maximum(n);
    }

    /**
     * Returns the maximum of a number and a {@link NDArray} element-wise.
     *
     * @param n the number to be compared
     * @param a the {@link NDArray} to be compared
     * @return the maximum of a number and a {@link NDArray} element-wise
     */
    public static NDArray maximum(Number n, NDArray a) {
        return maximum(a, n);
    }

    /**
     * Returns the maximum of {@link NDArray} a and {@link NDArray} b element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be compared
     * @param b the {@link NDArray} to be compared
     * @return the maximum of {@link NDArray} a and {@link NDArray} b element-wise
     */
    public static NDArray maximum(NDArray a, NDArray b) {
        return a.maximum(b);
    }

    /**
     * Returns the minimum of a {@link NDArray} and a number element-wise.
     *
     * @param a the {@link NDArray} to be compared
     * @param n the number to be compared
     * @return the minimum of a {@link NDArray} and a number element-wise
     */
    public static NDArray minimum(NDArray a, Number n) {
        return a.minimum(n);
    }

    /**
     * Returns the minimum of a number and a {@link NDArray} element-wise.
     *
     * @param n the number to be compared
     * @param a the {@link NDArray} to be compared
     * @return the minimum of a number and a {@link NDArray} element-wise
     */
    public static NDArray minimum(Number n, NDArray a) {
        return minimum(a, n);
    }

    /**
     * Returns the minimum of {@link NDArray} a and {@link NDArray} b element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be compared
     * @param b the {@link NDArray} to be compared
     * @return the minimum of {@link NDArray} a and {@link NDArray} b element-wise
     */
    public static NDArray minimum(NDArray a, NDArray b) {
        return a.minimum(b);
    }

    /**
     * Returns portion of the {@link NDArray} given the index boolean {@link NDArray} along first
     * axis.
     *
     * @param data the {@link NDArray} to operate on
     * @param index the boolean {@link NDArray} mask
     * @return the result {@link NDArray}
     */
    public static NDArray booleanMask(NDArray data, NDArray index) {
        return booleanMask(data, index, 0);
    }

    /**
     * Returns portion of the {@link NDArray} given the index boolean {@link NDArray} along given
     * axis.
     *
     * @param data the {@link NDArray} to operate on
     * @param index the boolean {@link NDArray} mask
     * @param axis an integer that represents the axis of {@link NDArray} to mask from
     * @return the result {@link NDArray}
     */
    public static NDArray booleanMask(NDArray data, NDArray index, int axis) {
        return data.booleanMask(index, axis);
    }

    ////////////////////////////////////////
    // Operators: Element Arithmetic
    ////////////////////////////////////////

    /**
     * Adds a number to the {@link NDArray} element-wise.
     *
     * @param a the {@link NDArray} to be added to
     * @param n the number to add
     * @return the result {@link NDArray}
     */
    public static NDArray add(NDArray a, Number n) {
        return a.add(n);
    }

    /**
     * Adds a {@link NDArray} to a number element-wise.
     *
     * @param n the number to be added to
     * @param a the {@link NDArray} to add
     * @return the result {@link NDArray}
     */
    public static NDArray add(Number n, NDArray a) {
        return a.add(n);
    }

    /**
     * Adds a {@link NDArray} to a {@link NDArray} element-wise.
     *
     * <p>The shapes of all of the {@link NDArray}s must be the same.
     *
     * @param arrays the {@link NDArray}s to add together
     * @return the result {@link NDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     * @throws IllegalArgumentException the shape of all inputs must be the same
     */
    public static NDArray add(NDArray... arrays) {
        checkInputs(arrays);
        if (arrays.length == 2) {
            return arrays[0].add(arrays[1]);
        }
        try (NDArray array = NDArrays.stack(new NDList(arrays))) {
            return array.sum(new int[] {0});
        }
    }

    /**
     * Subtracts a number from the {@link NDArray} element-wise.
     *
     * @param a the {@link NDArray} to be subtracted
     * @param n the number to subtract from
     * @return the result {@link NDArray}
     */
    public static NDArray sub(NDArray a, Number n) {
        return a.sub(n);
    }

    /**
     * Subtracts a {@link NDArray} from a number element-wise.
     *
     * @param n the number to be subtracted
     * @param a the {@link NDArray} to subtract from
     * @return the result {@link NDArray}
     */
    public static NDArray sub(Number n, NDArray a) {
        return a.getNDArrayInternal().rsub(n);
    }

    /**
     * Subtracts a {@link NDArray} from a {@link NDArray} element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be subtracted
     * @param b the {@link NDArray} to subtract from
     * @return the result {@link NDArray}
     */
    public static NDArray sub(NDArray a, NDArray b) {
        return a.sub(b);
    }

    /**
     * Multiplies the {@link NDArray} by a number element-wise.
     *
     * @param a the NDArray to be multiplied
     * @param n the number to multiply by
     * @return the result {@link NDArray}
     */
    public static NDArray mul(NDArray a, Number n) {
        return a.mul(n);
    }

    /**
     * Multiplies a number by a {@link NDArray} element-wise.
     *
     * @param n the number to be multiplied
     * @param a the {@link NDArray} to multiply by
     * @return the result {@link NDArray}
     */
    public static NDArray mul(Number n, NDArray a) {
        return a.mul(n);
    }

    /**
     * Multiplies all of the {@link NDArray}s together element-wise.
     *
     * <p>The shapes of all of the {@link NDArray}s must be the same.
     *
     * @param arrays the {@link NDArray}s to multiply together
     * @return the result {@link NDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     * @throws IllegalArgumentException the shape of all inputs must be the same
     */
    public static NDArray mul(NDArray... arrays) {
        checkInputs(arrays);
        if (arrays.length == 2) {
            return arrays[0].mul(arrays[1]);
        }
        try (NDArray array = NDArrays.stack(new NDList(arrays))) {
            return array.prod(new int[] {0});
        }
    }

    /**
     * Divides the {@link NDArray} by a number element-wise.
     *
     * @param a the {@link NDArray} to be be divided
     * @param n the number to divide by
     * @return the result {@link NDArray}
     */
    public static NDArray div(NDArray a, Number n) {
        return a.div(n);
    }

    /**
     * Divides a number by a {@link NDArray} element-wise.
     *
     * @param n the number to be be divided
     * @param a the {@link NDArray} to divide by
     * @return the result {@link NDArray}
     */
    public static NDArray div(Number n, NDArray a) {
        return a.getNDArrayInternal().rdiv(n);
    }

    /**
     * Divides a {@link NDArray} by a {@link NDArray} element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be be divided
     * @param b the {@link NDArray} to divide by
     * @return the result {@link NDArray}
     */
    public static NDArray div(NDArray a, NDArray b) {
        return a.div(b);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * @param a the dividend {@link NDArray}
     * @param n the divisor number
     * @return the result {@link NDArray}
     */
    public static NDArray mod(NDArray a, Number n) {
        return a.mod(n);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * @param n the dividend number
     * @param a the divisor {@link NDArray}
     * @return the result {@link NDArray}
     */
    public static NDArray mod(Number n, NDArray a) {
        return a.getNDArrayInternal().rmod(n);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * @param a the dividend NDArray
     * @param b the dividend NDArray
     * @return the result {@link NDArray}
     */
    public static NDArray mod(NDArray a, NDArray b) {
        return a.mod(b);
    }

    /**
     * Takes the power of the {@link NDArray} with a number element-wise.
     *
     * @param a the {@link NDArray} to be taken the power with
     * @param n the number to take the power with
     * @return the result {@link NDArray}
     */
    public static NDArray pow(NDArray a, Number n) {
        return a.pow(n);
    }

    /**
     * Takes the power of a number with a {@link NDArray} element-wise.
     *
     * @param n the number to be taken the power with
     * @param a the {@link NDArray} to take the power with
     * @return the result {@link NDArray}
     */
    public static NDArray pow(Number n, NDArray a) {
        return a.getNDArrayInternal().rpow(n);
    }

    /**
     * Takes the power of a {@link NDArray} with a {@link NDArray} element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be taken the power with
     * @param b the {@link NDArray} to take the power with
     * @return the result {@link NDArray}
     */
    public static NDArray pow(NDArray a, NDArray b) {
        return a.pow(b);
    }

    /**
     * Adds a number to the {@link NDArray} element-wise in place.
     *
     * @param a the {@link NDArray} to be added to
     * @param n the number to add
     * @return the result {@link NDArray}
     */
    public static NDArray addi(NDArray a, Number n) {
        return a.addi(n);
    }

    /**
     * Adds a {@link NDArray} to a number element-wise in place.
     *
     * @param a the number to be added to
     * @param n the {@link NDArray} to add
     * @return the result {@link NDArray}
     */
    public static NDArray addi(Number n, NDArray a) {
        return a.addi(n);
    }

    /**
     * Adds all of the {@link NDArray}s together element-wise in place.
     *
     * <p>The shapes of all of the {@link NDArray}s must be the same.
     *
     * @param arrays the {@link NDArray}s to add together
     * @return the result {@link NDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static NDArray addi(NDArray... arrays) {
        checkInputs(arrays);
        Arrays.stream(arrays).skip(1).forEachOrdered(array -> arrays[0].addi(array));
        return arrays[0];
    }

    /**
     * Subtracts a number from the {@link NDArray} element-wise in place.
     *
     * @param a the {@link NDArray} to be subtracted
     * @param n the number to subtract from
     * @return the result {@link NDArray}
     */
    public static NDArray subi(NDArray a, Number n) {
        return a.subi(n);
    }

    /**
     * Subtracts a {@link NDArray} from a number element-wise in place.
     *
     * @param n the number to be subtracted
     * @param a the {@link NDArray} to subtract from
     * @return the result {@link NDArray}
     */
    public static NDArray subi(Number n, NDArray a) {
        return a.getNDArrayInternal().rsubi(n);
    }

    /**
     * Subtracts a {@link NDArray} from a {@link NDArray} element-wise in place.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be subtracted
     * @param b the {@link NDArray} to subtract from
     * @return the result {@link NDArray}
     */
    public static NDArray subi(NDArray a, NDArray b) {
        return a.subi(b);
    }

    /**
     * Multiplies the {@link NDArray} by a number element-wise in place.
     *
     * @param a the NDArray to be multiplied
     * @param n the number to multiply by
     * @return the result {@link NDArray}
     */
    public static NDArray muli(NDArray a, Number n) {
        return a.muli(n);
    }

    /**
     * Multiplies a number by a {@link NDArray} element-wise.
     *
     * @param n the number to multiply by
     * @param a the {@link NDArray} to multiply by
     * @return the result {@link NDArray}
     */
    public static NDArray muli(Number n, NDArray a) {
        return a.muli(n);
    }

    /**
     * Multiplies all of the {@link NDArray}s together element-wise in place.
     *
     * <p>The shapes of all of the {@link NDArray}s must be the same.
     *
     * @param arrays the {@link NDArray}s to multiply together
     * @return the result {@link NDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static NDArray muli(NDArray... arrays) {
        checkInputs(arrays);
        Arrays.stream(arrays).skip(1).forEachOrdered(array -> arrays[0].muli(array));
        return arrays[0];
    }

    /**
     * Divides a number by a {@link NDArray} element-wise in place.
     *
     * @param a the {@link NDArray} to be be divided
     * @param n the number to divide by
     * @return the result {@link NDArray}
     */
    public static NDArray divi(NDArray a, Number n) {
        return a.divi(n);
    }

    /**
     * Divides a number by a {@link NDArray} element-wise.
     *
     * @param n the number to be be divided
     * @param a the {@link NDArray} to divide by
     * @return the result {@link NDArray}
     */
    public static NDArray divi(Number n, NDArray a) {
        return a.getNDArrayInternal().rdivi(n);
    }

    /**
     * Divides a {@link NDArray} by a {@link NDArray} element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be be divided
     * @param b the {@link NDArray} to divide by
     * @return the result {@link NDArray}
     */
    public static NDArray divi(NDArray a, NDArray b) {
        return a.divi(b);
    }

    /**
     * Returns element-wise remainder of division in place.
     *
     * @param a the dividend {@link NDArray}
     * @param n the divisor number
     * @return the result {@link NDArray}
     */
    public static NDArray modi(NDArray a, Number n) {
        return a.modi(n);
    }

    /**
     * Returns element-wise remainder of division in place.
     *
     * @param n the dividend number
     * @param a the divisor {@link NDArray}
     * @return the result {@link NDArray}
     */
    public static NDArray modi(Number n, NDArray a) {
        return a.getNDArrayInternal().rmodi(n);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * @param a the dividend NDArray
     * @param b the dividend NDArray
     * @return the result {@link NDArray}
     */
    public static NDArray modi(NDArray a, NDArray b) {
        return a.modi(b);
    }

    /**
     * Takes the power of the {@link NDArray} with a number element-wise in place.
     *
     * @param a the {@link NDArray} to be taken the power with
     * @param n the number to take the power with
     * @return the result {@link NDArray}
     */
    public static NDArray powi(NDArray a, Number n) {
        return a.powi(n);
    }

    /**
     * Takes the power of a number with a {@link NDArray} element-wise in place.
     *
     * @param n the number to be taken the power with
     * @param a the {@link NDArray} to take the power with
     * @return the result {@link NDArray}
     */
    public static NDArray powi(Number n, NDArray a) {
        return a.getNDArrayInternal().rpowi(n);
    }

    /**
     * Takes the power of a {@link NDArray} with a {@link NDArray} element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to be taken the power with
     * @param b the {@link NDArray} to take the power with
     * @return the result {@link NDArray}
     */
    public static NDArray powi(NDArray a, NDArray b) {
        return a.powi(b);
    }

    /**
     * Dot product of {@link NDArray} a and {@link NDArray} b.
     *
     * <ul>
     *   <li>If both the {@link NDArray} and the other {@link NDArray} are 1-D {@link NDArray}s, it
     *       is inner product of vectors (without complex conjugation).
     *   <li>If both the {@link NDArray} and the other {@link NDArray} are 2-D {@link NDArray}s, it
     *       is matrix multiplication.
     *   <li>If either the {@link NDArray} or the other {@link NDArray} is 0-D {@link NDArray}
     *       (scalar), it is equivalent to mul.
     *   <li>If the {@link NDArray} is N-D {@link NDArray} and the other {@link NDArray} is 1-D
     *       {@link NDArray}, it is a sum product over the last axis of those.
     *   <li>If the {@link NDArray} is N-D {@link NDArray} and the other {@link NDArray} is M-D
     *       {@link NDArray}(where M&gt;&#61;2), it is a sum product over the last axis of this
     *       {@link NDArray} and the second-to-last axis of the other {@link NDArray}
     * </ul>
     *
     * @param a the {@link NDArray} to perform dot product with
     * @param b the {@link NDArray} to perform dot product with
     * @return the result {@link NDArray}
     */
    public static NDArray dot(NDArray a, NDArray b) {
        return a.dot(b);
    }

    /**
     * Joins a sequence of {@link NDArray}s in {@link NDList} along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last
     * dimension.
     *
     * @param arrays the input {@link NDList}. Each {@link NDArray} in the {@link NDList} must have
     *     the same shape as the {@link NDArray}
     * @param axis the axis in the result {@link NDArray} along which the input {@link NDList} are
     *     stacked
     * @return the result {@link NDArray}. The stacked {@link NDArray} has one more dimension than
     *     the the {@link NDArray}
     */
    public static NDArray stack(NDList arrays, int axis) {
        NDArray array = arrays.head();
        return array.getNDArrayInternal().stack(arrays.subNDList(1), axis);
    }

    /**
     * Joins a sequence of {@link NDArray}s in {@link NDList} along first axis.
     *
     * @param arrays the input {@link NDList}. Each {@link NDArray} in the {@link NDList} must have
     *     the same shape as the {@link NDArray}
     * @return the result {@link NDArray}. The stacked {@link NDArray} has one more dimension than
     *     the {@link NDArray}s in {@link NDList}
     */
    public static NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a {@link NDList} along an existing axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@link NDArray}, except in
     *     the dimension corresponding to axis
     * @param axis the axis along which the {@link NDList} will be joined
     * @return the concatenated {@link NDArray}
     */
    public static NDArray concat(NDList arrays, int axis) {
        NDArray array = arrays.head();
        return array.getNDArrayInternal().concat(arrays.subNDList(1), axis);
    }

    /**
     * Joins a {@link NDList} along first axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@link NDArray}, except in
     *     the dimension corresponding to axis
     * @return the concatenated {@link NDArray}
     */
    public static NDArray concat(NDList arrays) {
        return concat(arrays, 0);
    }

    /**
     * Returns the truth value of {@link NDArray} a AND {@link NDArray} b element-wise.
     *
     * <p>The shapes of {@link NDArray} a and {@link NDArray} b must be broadcastable.
     *
     * @param a the {@link NDArray} to operate on
     * @param b the {@link NDArray} to operate on
     * @return the boolean {@link NDArray} of the logical AND operation applied to the elements of
     *     the {@link NDArray} a and {@link NDArray} b
     */
    public static NDArray logicalAnd(NDArray a, NDArray b) {
        return a.logicalAnd(b);
    }

    /**
     * Computes the truth value of {@link NDArray} a AND {@link NDArray} b element-wise.
     *
     * @param a the {@link NDArray} to operate on
     * @param b the {@link NDArray} to operate on
     * @return the boolean {@link NDArray} of the logical AND operation applied to the elements of
     *     the {@link NDArray} a and {@link NDArray} b
     */
    public static NDArray logicalOr(NDArray a, NDArray b) {
        return a.logicalOr(b);
    }

    /**
     * Computes the truth value of {@link NDArray} a AND {@link NDArray} b element-wise.
     *
     * @param a the {@link NDArray} to operate on
     * @param b the {@link NDArray} to operate on
     * @return the boolean {@link NDArray} of the logical XOR operation applied to the elements of
     *     the {@link NDArray} a and {@link NDArray} b
     */
    public static NDArray logicalXor(NDArray a, NDArray b) {
        return a.logicalXor(b);
    }
}
