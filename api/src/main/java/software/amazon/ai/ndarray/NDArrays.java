package software.amazon.ai.ndarray;

import java.util.Arrays;

/** This class contains various methods for manipulating NDArrays. */
public final class NDArrays {

    private NDArrays() {}

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return Returns the result of the addition
     */
    public static NDArray add(NDArray a, Number n) {
        return a.add(n);
    }

    /**
     * In place Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return Returns the result of the addition
     */
    public static NDArray addi(NDArray a, Number n) {
        return a.addi(n);
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return Returns the result of the addition
     */
    public static NDArray add(Number n, NDArray a) {
        return a.add(n);
    }

    /**
     * Adds two {@link NDArray}s with broadcasting.
     *
     * @param a the left NDArray
     * @param b the right NDArray
     * @return Returns the result of the addition
     */
    public static NDArray add(NDArray a, NDArray b) {
        return a.add(b);
    }

    /**
     * Adds two {@link NDArray}s with broadcasting.
     *
     * @param a the left NDArray
     * @param b the right NDArray
     * @return Returns the result of the addition
     */
    public static NDArray addi(NDArray a, NDArray b) {
        return a.addi(b);
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
     * Returns the boolean true iff all elements in both the NDArrays are equal.
     *
     * @param a the ndarray to compare.
     * @param b the ndarray to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    public static boolean equals(NDArray a, NDArray b) {
        return a.contentEquals(b);
    }

    /**
     * Returns the boolean true iff all elements in the NDArray is equal to the Number
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
    public NDArray gt(NDArray a, NDArray b) {
        return a.gt(b);
    }

    /**
     * Returns the binary ndarray for "Greater or Equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the ndarray to be compared against
     * @return the binary ndarray for "Greater Than" comparison.
     */
    public NDArray gte(NDArray a, NDArray b) {
        return a.gte(b);
    }

    /**
     * Returns binary ndarray for "Greater or equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return binary ndarray for "Greater or equals" comparison.
     */
    public NDArray gte(NDArray a, Number b) {
        return a.gte(b);
    }

    /**
     * Returns the binary ndarray for "Less or equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return the binary ndarray for "Less or equals" comparison.
     */
    public NDArray lte(NDArray a, Number b) {
        return a.lte(b);
    }

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return the binary ndarray for "Less" comparison.
     */
    public NDArray lt(NDArray a, Number b) {
        return a.lt(b);
    }

    /**
     * Returns the binary ndarray for "Lesser or equals" comparison.
     *
     * @param a ndarray to be compared
     * @param b the ndarray to be compared against
     * @return the binary ndarray for "Less" comparison.
     */
    public NDArray lte(NDArray a, NDArray b) {
        return a.lte(b);
    }

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param a ndarray to be compared
     * @param b the ndarray to be compared against
     * @return the binary ndarray for "Less" comparison.
     */
    public NDArray lt(NDArray a, NDArray b) {
        return a.lt(b);
    }

    /**
     * Returns the binary ndarray for "Greater" comparison.
     *
     * @param a ndarray to be compared
     * @param b the number to be compared against
     * @return the binary ndarray for "Greater" comparison.
     */
    public NDArray gt(NDArray a, Number b) {
        return a.gt(b);
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
     * Join a sequence of {@link NDArray} along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDArray[] arrays) {
        return stack(arrays, 0);
    }

    /**
     * Join a sequence of {@link NDArray} in NDList along a new axis.
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
     * Join a sequence of {@link NDArray} in NDList along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray stack(NDList arrays) {
        return stack(arrays.toArray(), 0);
    }

    /**
     * Join a sequence of {@link NDArray} along an existing axis.
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
     * Join a sequence of {@link NDArray} along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDArray[] arrays) {
        return concat(arrays, 0);
    }

    /**
     * Join a sequence of {@link NDArray} in NDList along existing axis.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDList arrays, int axis) {
        return concat(arrays.toArray(), axis);
    }

    /**
     * Join a sequence of {@link NDArray} in NDList along axis 0.
     *
     * @param arrays input NDList. each {@link NDArray} must have the same shape.
     * @return {@link NDArray}. The stacked array has one more dimension than the input arrays.
     */
    public static NDArray concat(NDList arrays) {
        return concat(arrays.toArray(), 0);
    }
}
