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

import ai.djl.Device;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * An interface representing an n-dimensional array.
 *
 * <p>NDArray is the core data structure for all mathematical computations. An NDArray represents a
 * multidimensional, fixed-size homogeneous array. It has very similar behaviour to the Numpy python
 * package with the addition of efficient computing.
 */
public interface NDArray extends AutoCloseable {

    /**
     * Returns the {@link NDManager} used to create the {@code NDArray}.
     *
     * @return the {@link NDManager} used to create the {@code NDArray}
     */
    NDManager getManager();

    /**
     * Returns the name of the {@code NDArray}.
     *
     * @return the name of the {@code NDArray}
     */
    String getName();

    /**
     * Sets name of the {@code NDArray}.
     *
     * @param name the name of the {@code NDArray}
     */
    void setName(String name);

    /**
     * Returns unique identifier of the {@code NDArray}.
     *
     * @return unique identifier of the {@code NDArray}
     */
    String getUid();

    /**
     * Returns the {@link DataType} of the {@code NDArray}.
     *
     * <p>{@link DataType} is a definition of the precision level of the {@code NDArray}. All values
     * inside the same NDArray would have the same data type.
     *
     * @return the {@link DataType} of the {@code NDArray}
     */
    DataType getDataType();

    /**
     * Returns the {@link Device} of the {@code NDArray}.
     *
     * <p>{@link Device} class contains the information where this NDArray stored in memory, like
     * CPU/GPU.
     *
     * @return the {@link Device} of the {@code NDArray}
     */
    Device getDevice();

    /**
     * Returns the {@link Shape} of the {@code NDArray}.
     *
     * <p>{@link Shape} defines how this NDArray is represented multi-dimensionally.
     *
     * @return the {@link Shape} of the {@code NDArray}
     */
    Shape getShape();

    /**
     * Returns the {@link SparseFormat} of the {@code NDArray}.
     *
     * @return the {@link SparseFormat} of the {@code NDArray}
     */
    SparseFormat getSparseFormat();

    /**
     * Returns {@code true} if this {@code NDArray} is a {@link SparseNDArray}.
     *
     * @return {@code true} if this {@code NDArray} is a {@link SparseNDArray}
     */
    default boolean isSparse() {
        return getSparseFormat() != SparseFormat.DENSE;
    }

    /**
     * Returns {@code true} if this {@code NDArray} is a scalar with 0 dimension shape.
     *
     * @return {@code true} if this {@code NDArray} is a scalar
     */
    default boolean isScalar() {
        return getShape().isScalar();
    }

    /**
     * Attaches this {@code NDArray} to the specified {@link NDManager}.
     *
     * <p>Attached resource will be closed when the manager is closed.
     *
     * @param manager the {@link NDManager} to be attached
     * @see NDManager
     */
    default void attach(NDManager manager) {
        detach();
        getManager().attach(getUid(), manager);
    }

    /**
     * Detaches this {@code NDArray} from current {@link NDManager}'s lifecycle.
     *
     * <p>This NDArray becomes un-managed, it is the user's responsibility to close the NDArray.
     * Failure to close the resource might cause your machine to run out of native memory.
     *
     * @see NDManager
     */
    default void detach() {
        getManager().detach(getUid());
    }

    /**
     * Converts the {@code NDArray} to a different {@link Device}.
     *
     * @param device the {@link Device} to be set
     * @param copy set {@code true} if you want to return a copy of the Existing {@code NDArray}
     * @return the result {@code NDArray} with the new {@link Device}
     */
    NDArray asInDevice(Device device, boolean copy);

    /**
     * Converts the NDArray to a different {@link DataType}.
     *
     * @param dtype the {@link DataType} to be set
     * @param copy set {@code true} if you want to return a copy of the Existing {@code NDArray}
     * @return the result {@code NDArray} with the new {@link DataType}
     */
    NDArray asType(DataType dtype, boolean copy);

    /**
     * Converts the array into a 2-D {@link Matrix}.
     *
     * @return this NDArray as Matrix
     * @throws IllegalStateException thrown if the NDArray is not a 2-D matrix
     */
    Matrix asMatrix();

    /**
     * Attaches a gradient {@code NDArray} to this {@code NDArray} and marks it so {@code backward}
     * can compute the gradient with respect to it.
     */
    void attachGradient();

    /**
     * Returns the gradient {@code NDArray} attached to this {@code NDArray}.
     *
     * @return the gradient {@code NDArray}
     * @throws NullPointerException when gradient is not initialized
     */
    NDArray getGradient();

    /**
     * Returns the size along a specified axis.
     *
     * @param axis the axis to return the size for
     * @return the size of the array along the specified axis
     */
    default long size(int axis) {
        return getShape().size(axis);
    }

    /**
     * Returns the total number of elements in the {@code NDArray}.
     *
     * @return the number of elements in the {@code NDArray}
     */
    default long size() {
        return getShape().size();
    }

    /**
     * Converts this {@code NDArray} to a double array.
     *
     * @return a double array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    default double[] toDoubleArray() {
        if (getDataType() != DataType.FLOAT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required double" + " Actual " + getDataType());
        }
        DoubleBuffer db = toByteBuffer().asDoubleBuffer();
        double[] ret = new double[db.remaining()];
        db.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a float array.
     *
     * @return a float array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    default float[] toFloatArray() {
        if (getDataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required float, Actual " + getDataType());
        }
        FloatBuffer fb = toByteBuffer().asFloatBuffer();
        float[] ret = new float[fb.remaining()];
        fb.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to an int array.
     *
     * @return an int array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray}} mismatches
     */
    default int[] toIntArray() {
        if (getDataType() != DataType.INT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required int" + " Actual " + getDataType());
        }
        IntBuffer ib = toByteBuffer().asIntBuffer();
        int[] ret = new int[ib.remaining()];
        ib.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a long array.
     *
     * @return a long array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    default long[] toLongArray() {
        if (getDataType() != DataType.INT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required long" + " Actual " + getDataType());
        }
        LongBuffer lb = toByteBuffer().asLongBuffer();
        long[] ret = new long[lb.remaining()];
        lb.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a byte array.
     *
     * @return a byte array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    default byte[] toByteArray() {
        ByteBuffer bb = toByteBuffer();
        if (bb.hasArray()) {
            return bb.array();
        }
        byte[] buf = new byte[bb.remaining()];
        bb.get(buf);
        return buf;
    }

    /**
     * Converts this {@code NDArray} to a uint8 array.
     *
     * @return a uint8 array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    default int[] toUint8Array() {
        ByteBuffer bb = toByteBuffer();
        int[] buf = new int[bb.remaining()];
        for (int i = 0; i < buf.length; ++i) {
            buf[i] = bb.get() & 0xff;
        }
        return buf;
    }

    /**
     * Converts this {@code NDArray} to a boolean array.
     *
     * @return a boolean array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    default boolean[] toBoolArray() {
        if (getDataType() != DataType.BOOLEAN) {
            throw new IllegalStateException(
                    "DataType mismatch, Required boolean" + " Actual " + getDataType());
        }
        ByteBuffer bb = toByteBuffer();
        boolean[] ret = new boolean[bb.remaining()];
        for (int i = 0; i < ret.length; ++i) {
            ret[i] = bb.get() != 0;
        }
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a Number array based on its {@link DataType}.
     *
     * @return a Number array
     */
    default Number[] toArray() {
        switch (getDataType()) {
            case FLOAT32:
                float[] floatArray = toFloatArray();
                return IntStream.range(0, floatArray.length)
                        .mapToObj(i -> floatArray[i])
                        .toArray(Number[]::new);
            case FLOAT64:
                return Arrays.stream(toDoubleArray()).boxed().toArray(Double[]::new);
            case INT32:
                return Arrays.stream(toIntArray()).boxed().toArray(Integer[]::new);
            case INT64:
                return Arrays.stream(toLongArray()).boxed().toArray(Long[]::new);
            case BOOLEAN:
            case INT8:
                ByteBuffer bb = toByteBuffer();
                Byte[] ret = new Byte[bb.remaining()];
                for (int i = 0; i < ret.length; ++i) {
                    ret[i] = bb.get();
                }
                return ret;
            case UINT8:
                return Arrays.stream(toUint8Array()).boxed().toArray(Integer[]::new);
            default:
                throw new IllegalStateException("Unsupported DataType: " + getDataType());
        }
    }

    /**
     * Converts this {@code NDArray} to a ByteBuffer.
     *
     * @return a ByteBuffer
     */
    ByteBuffer toByteBuffer();

    /**
     * Sets the {@code NDArray} value from {@link Buffer}.
     *
     * @param data the input buffered data
     */
    void set(Buffer data);

    /**
     * Sets the {@code NDArray} value from an array of floats.
     *
     * @param data the array of floats to set
     */
    default void set(float[] data) {
        set(FloatBuffer.wrap(data));
    }

    /**
     * Sets the {@code NDArray} value from an array of ints.
     *
     * @param data the array of integers to set
     */
    default void set(int[] data) {
        set(IntBuffer.wrap(data));
    }

    /**
     * Sets the {@code NDArray} value from an array of doubles.
     *
     * @param data the array of doubles to set
     */
    default void set(double[] data) {
        set(DoubleBuffer.wrap(data));
    }

    /**
     * Sets the {@code NDArray} value from an array of longs.
     *
     * @param data the array of longs to set
     */
    default void set(long[] data) {
        set(LongBuffer.wrap(data));
    }

    /**
     * Sets the {@code NDArray} value from an array of bytes.
     *
     * @param data the array of bytes to set
     */
    default void set(byte[] data) {
        set(ByteBuffer.wrap(data));
    }

    /**
     * Sets the specified index in the {@code NDArray} with the given values.
     *
     * @param index the locations to update
     * @param value the value to replace with. Can broadcast if given smaller dimensions than the
     *     index
     */
    void set(NDIndex index, NDArray value);

    /**
     * Sets the specified index in the {@code NDArray} with the given value.
     *
     * @param index the locations to update
     * @param value the value to replace with
     */
    void set(NDIndex index, Number value);

    /**
     * Sets the specified index in the {@code NDArray} with the given value.
     *
     * @param index the single index to update
     * @param value the value to replace with
     * @throws IllegalArgumentException thrown if the index does not correspond to a single element
     */
    void setScalar(NDIndex index, Number value);

    /**
     * Returns a partial {@code NDArray}.
     *
     * @param index the section of the {@code NDArray} to return
     * @return the partial {@code NDArray}
     */
    NDArray get(NDIndex index);

    /**
     * Returns a partial {@code NDArray}.
     *
     * @param indices the indices used to indicate what to get
     * @return the partial {@code NDArray}
     * @see NDIndex#NDIndex(String)
     */
    default NDArray get(String indices) {
        return get(new NDIndex(indices));
    }

    /**
     * Returns a partial {@code NDArray}.
     *
     * @param indices the indices with each index corresponding to the dimensions and negative
     *     indices starting from the end
     * @return the partial {@code NDArray}
     */
    default NDArray get(long... indices) {
        return get(new NDIndex(indices));
    }

    /**
     * Returns a partial {@code NDArray}.
     *
     * @param index the boolean {@code NDArray} that indicates what to get
     * @return the partial {@code NDArray}
     */
    default NDArray get(NDArray index) {
        return get(new NDIndex().addBooleanIndex(index));
    }

    /**
     * Returns a scalar {@code NDArray} corresponding to a single element.
     *
     * @param indices the indices of the scalar to return. Must return only a single element
     * @return a scalar {@code NDArray} corresponding to the element
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default NDArray getScalar(long... indices) {
        NDArray value = get(new NDIndex(indices));
        if (value.size() != 1) {
            throw new IllegalArgumentException("The supplied Index does not produce a scalar");
        }
        return value;
    }

    /**
     * Returns a long element from the {@code NDArray}.
     *
     * @param indices the indices of the long element to return
     * @return the element in the specified index as a long
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default long getLong(long... indices) {
        return getScalar(indices).toLongArray()[0];
    }

    /**
     * Returns a double element from the {@code NDArray}.
     *
     * @param indices the indices of the double element to return
     * @return the element in the specified index as a double
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default double getDouble(long... indices) {
        return getScalar(indices).toDoubleArray()[0];
    }

    /**
     * Returns a float element from the {@code NDArray}.
     *
     * @param indices the indices of the long element to return
     * @return the element in the specified index as a float
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default float getFloat(long... indices) {
        return getScalar(indices).toFloatArray()[0];
    }

    /**
     * Returns an int element from the {@code NDArray}.
     *
     * @param indices the indices of the int element to return
     * @return the element in the specified index as an integer
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default int getInt(long... indices) {
        return getScalar(indices).toIntArray()[0];
    }

    /**
     * Returns an byte element from the {@code NDArray}.
     *
     * @param indices the indices of the byte element to return
     * @return the element in the specified index as a byte
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default byte getByte(long... indices) {
        return getScalar(indices).toByteArray()[0];
    }

    /**
     * Returns an integer element from the {@code NDArray} that represent unsigned integer with 8
     * bits.
     *
     * @param indices the indices of the unsigned 8 bits integer element to return
     * @return the element in the specified index as a uint8
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default int getUint8(long... indices) {
        return getByte(indices) & 0xff;
    }

    /**
     * Deep-copies the current {@code NDArray} to the one passed in.
     *
     * @param array the {@code NDArray} prepared to be copied to
     */
    void copyTo(NDArray array);

    /**
     * Creates a new {@code NDArray} whose content is a copy of this {@code NDArray}'s content.
     *
     * @return the new {@code NDArray}
     */
    default NDArray slice() {
        // TODO: MXNet doesn't support view, return a copy for now
        return duplicate();
    }

    /**
     * Returns a copy of this {@code NDArray}.
     *
     * @return a copy of this {@code NDArray}
     */
    default NDArray duplicate() {
        NDArray array = getManager().create(getShape(), getDataType(), getDevice());
        array.setName(getName());
        copyTo(array);
        return array;
    }

    /**
     * Returns portion of {@code NDArray} given the index boolean {@code NDArray} on first axis.
     *
     * @param index boolean {@code NDArray} mask
     * @return the result {@code NDArray}
     */
    default NDArray booleanMask(NDArray index) {
        return booleanMask(index, 0);
    }

    /**
     * Returns portion of {@code NDArray} given the index boolean {@code NDArray} on first axis.
     *
     * @param index boolean {@code NDArray} mask
     * @param axis an integer that represents the axis of {@code NDArray} to mask from
     * @return the result {@code NDArray}
     */
    NDArray booleanMask(NDArray index, int axis);

    /**
     * Returns an {@code NDArray} of zeros with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input {@code NDArray}.
     *
     * @return a {@code NDArray} filled with zeros
     */
    NDArray zerosLike();

    /**
     * Returns an {@code NDArray} of ones with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input {@code NDArray}.
     *
     * @return a {@code NDArray} filled with ones
     */
    NDArray onesLike();

    /**
     * Returns an uninitialized {@code NDArray} with the same {@link Shape}, {@link DataType} and
     * {@link SparseFormat} as the input {@code NDArray}.
     *
     * @return the result {@code NDArray}
     */
    default NDArray like() {
        return getManager().create(getShape());
    }

    ////////////////////////////////////////
    ////////////////////////////////////////
    // Operators
    ////////////////////////////////////////
    ////////////////////////////////////////

    ////////////////////////////////////////
    // Operators: Element Comparison
    ////////////////////////////////////////

    /**
     * Returns true if all elements in the {@code NDArray} are equal to the {@code Number}.
     *
     * @param number the number to compare
     * @return the boolean result
     */
    boolean contentEquals(Number number);

    /**
     * Returns true if all elements in the {@code NDArray} are equal to the other {@code NDArray}.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean result
     */
    boolean contentEquals(NDArray other);

    /**
     * Returns true if two {@code NDArray}s are element-wise equal within a tolerance.
     *
     * @param other the {@code NDArray} to compare with
     * @return the boolean result
     */
    default boolean allClose(NDArray other) {
        return allClose(other, 1e-5, 1e-08, false);
    }

    /**
     * Returns true if two {@code NDArray} are element-wise equal within a tolerance.
     *
     * @param other the {@code NDArray} to compare with
     * @param rtol the relative tolerance parameter
     * @param atol the absolute tolerance parameter
     * @param equalNan whether to compare NaN’s as equal. If true, NaN’s in this {@code NDArray}
     *     will be considered equal to NaN’s in the other {@code NDArray}
     * @return the boolean result
     */
    default boolean allClose(NDArray other, double rtol, double atol, boolean equalNan) {
        if (!getShape().equals(other.getShape())) {
            return false;
        }
        Number[] actualDoubleArray = toArray();
        Number[] expectedDoubleArray = other.toArray();
        for (int i = 0; i < actualDoubleArray.length; i++) {
            double a = actualDoubleArray[i].doubleValue();
            double b = expectedDoubleArray[i].doubleValue();
            // handle NaN
            if (equalNan && Double.isNaN(a) && Double.isNaN(b)) {
                continue;
            }
            if (Double.isNaN(a)
                    || Double.isNaN(b)
                    || (Math.abs(a - b) > (atol + rtol * Math.abs(b)))) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Equals" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Equals" comparison
     */
    NDArray eq(Number other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Equals" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Equals" comparison
     */
    NDArray eq(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Not equals" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Not equals" comparison
     */
    NDArray neq(Number other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Not equals" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Not equals" comparison
     */
    NDArray neq(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Greater" comparison
     */
    NDArray gt(Number other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater Than" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wis "Greater Than" comparison
     */
    NDArray gt(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater or equals" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Greater or equals" comparison
     */
    NDArray gte(Number other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater or equals" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for "Greater or equals" comparison
     */
    NDArray gte(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Less" comparison
     */
    NDArray lt(Number other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Less" comparison
     */
    NDArray lt(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less or equals" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Less or equals" comparison
     */
    NDArray lte(Number other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less or equals" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Less or equals" comparison
     */
    NDArray lte(NDArray other);

    // TODO where operator is not compliant with numpy
    /**
     * Return the elements, either from this NDArray or other, depending on the condition.
     *
     * <p>Given three NDArrays, condition, this, and other, return an NDArray with the elements from
     * this or other, depending on whether the elements from condition are true or false. The other
     * array must have the same shape as this. If condition has the same shape as this, each element
     * in the output array is from this if the corresponding element in the condition is true, and
     * from other if false.
     *
     * <p>If condition does not have the same shape as this, it must be a 1D array whose size is the
     * same as this array's first dimension size. Each row of the output array is from this array's
     * row if the corresponding element from condition is true, and from other’s row if false.
     *
     * <p>Note that all non-zero values are interpreted as True in condition.
     *
     * @param condition the condition array
     * @param other the other NDArray
     * @return the result NDArray
     */
    NDArray where(NDArray condition, NDArray other);

    ////////////////////////////////////////
    // Operators: Element Arithmetic
    ////////////////////////////////////////

    /**
     * Adds a number to each element of the array.
     *
     * @param n the number to add
     * @return the result of the addition
     */
    NDArray add(Number n);

    /**
     * Adds (broadcasting) another NDArray to this {@code NDArray}.
     *
     * @param others the other NDArrays to add
     * @return the result of the addition
     */
    NDArray add(NDArray... others);

    /**
     * Applies scalar subtraction on an array (copied).
     *
     * @param n the number to subtract by
     * @return a copy of this array after applying subtraction operation
     */
    NDArray sub(Number n);

    /**
     * Applies copy subtraction of two NDArrays.
     *
     * @param other the second NDArray to subtract
     * @return the result of the subtraction
     */
    NDArray sub(NDArray other);

    /**
     * Applies scalar multiplication on an array (copy).
     *
     * @param n the number to multiply by
     * @return a copy of this NDArray multiplied by the given number
     */
    NDArray mul(Number n);

    /**
     * Applies element-wise multiplication of other NDArrays to this NDArray.
     *
     * @param others the other NDArrays to multiply with
     * @return the result of the multiplication
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray mul(NDArray... others);

    /**
     * Divides an array by a number.
     *
     * @param n the number to divide values by
     * @return a copy of the array after division
     */
    NDArray div(Number n);

    /**
     * Applies element-wise division of two NDArrays.
     *
     * @param other the second NDArray to divide
     * @return the result of the divide
     */
    NDArray div(NDArray other);

    /**
     * Returns element-wise remainder of division.
     *
     * <p>NDArray nd = manager.create(new float[] {-3, -5}, null, new Shape(2)); nd.mod(-2) //
     * return [-1, -1]
     *
     * @param n the divisor number
     * @return a copy of {@code NDArray} after division
     */
    NDArray mod(Number n);

    /**
     * Returns element-wise remainder of division.
     *
     * @param other the second NDArray to divide by
     * @return the result of the divide
     */
    NDArray mod(NDArray other);

    /**
     * Raises the power of each element in the {@code NDArray}.
     *
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    NDArray pow(Number n);

    /**
     * Raises the power of each element in the NDArray by the corresponding element in the other
     * {@code NDArray}.
     *
     * @param other the NDArray by which the raise the power by
     * @return the result {@code NDArray}
     */
    NDArray pow(NDArray other);

    /**
     * Adds a number to each element of the array in place.
     *
     * @param n the number to add
     * @return the result of the addition
     */
    NDArray addi(Number n);

    /**
     * Adds (broadcasting) another NDArray to this NDArray in place.
     *
     * @param others the other NDArrays to add
     * @return the result of the addition
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray addi(NDArray... others);

    /**
     * Applies in place scalar subtraction of an array.
     *
     * @param n the number to subtract
     * @return this array after applying the subtraction operation
     */
    NDArray subi(Number n);

    /**
     * Performs in place (element wise) subtraction of two NDArrays.
     *
     * @param other the second NDArray to subtract
     * @return the result of the subtraction
     */
    NDArray subi(NDArray other);

    /**
     * Performs in place scalar multiplication of an array.
     *
     * @param n the number to multiply by
     * @return this array after applying scalar multiplication
     */
    NDArray muli(Number n);

    /**
     * Performs element-wise multiplication of other NDArrays to this NDArray in place.
     *
     * @param others the other NDArrays to multiply with
     * @return the result of the multiplication
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray muli(NDArray... others);

    /**
     * Performs in place scalar division of an array.
     *
     * @param n the number to divide values by
     * @return this array after applying division operation
     */
    NDArray divi(Number n);

    /**
     * Performs in place element-wise division of two NDArrays.
     *
     * @param other the second NDArray to divide by
     * @return the result of the divide
     */
    NDArray divi(NDArray other);

    /**
     * Returns element-wise remainder of division.
     *
     * @param n the divisor number
     * @return a copy of {@code NDArray} after division
     */
    NDArray modi(Number n);

    /**
     * Returns in place element-wise remainder of division.
     *
     * @param other the second NDArray to divide
     * @return the result of the divide
     */
    NDArray modi(NDArray other);

    /**
     * Raises the power of each element in the NDArray in-place.
     *
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    NDArray powi(Number n);

    /**
     * Raises the power of each element in the NDArray by the corresponding element in the other
     * NDArray in-place.
     *
     * @param other the NDArray by which the raise the power by
     * @return the result {@code NDArray}
     */
    NDArray powi(NDArray other);

    ////////////////////////////////////////
    // Operators: Basic Numeric
    ////////////////////////////////////////

    /**
     * Returns the NDArray negative (cloned).
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5);
     * jshell&gt; array.neg();
     * ND: (5) cpu(0) float32
     * [-0., -1., -2., -3., -4.]
     * </pre>
     *
     * @return a copy of the array with all values negated
     */
    NDArray neg();

    /**
     * Sets the negative version of this NDArray in place.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5);
     * jshell&gt; array.negi();
     * jshell&gt; array;
     * ND: (5) cpu(0) float32
     * [-0., -1., -2., -3., -4.]
     * </pre>
     *
     * @return this array with all values negated
     */
    NDArray negi();

    /**
     * Calculates the absolute value element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1f, -2f});
     * jshell&gt; array.abs();
     * ND: (2) cpu(0) float32
     * [1., 2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray abs();

    /**
     * Returns the element-wise square of the input.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, -3f});
     * jshell&gt; array.square();
     * ND: (2) cpu(0) float32
     * [4., 9.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray square();

    /**
     * Returns the cube-root of an array, element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 8f, 27f});
     * jshell&gt; array.cbrt();
     * ND: (3) cpu(0) float32
     * [1., 2., 3.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray cbrt();

    /**
     * Returns the floor of the input, element-wise. The floor of the scalar x is the largest
     * integer i, such that i &lt;= x. It is often denoted as \lfloor x \rfloor.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.floor();
     * ND: (7) cpu(0) float32
     * [-2., -2., -1.,  0.,  1.,  1.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray floor();

    /**
     * Returns the ceiling of the input, element-wise. The ceil of the scalar x is the smallest
     * integer i, such that i &gt;= x. It is often denoted as \lceil x \rceil.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.ceil();
     * ND: (7) cpu(0) float32
     * [-1., -1., -0.,  1.,  2.,  2.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray ceil();

    /**
     * Rounds elements of the array to the nearest integer.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.round();
     * ND: (7) cpu(0) float32
     * [-2., -2., -0.,  0.,  2.,  2.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray round();

    /**
     * Returns the element-wise truncated value of the input.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.trunc();
     * ND: (7) cpu(0) float32
     * [-1., -1., -0.,  0.,  1.,  1.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray trunc();

    /**
     * Returns element-wise exponential value of the input.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 2.5f});
     * jshell&gt; array.exp();
     * ND: (2) cpu(0) float32
     * [ 1.    , 12.1825]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray exp();

    /**
     * Returns element-wise Natural logarithmic value of the input.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 2.5f});
     * jshell&gt; array.log();
     * ND: (2) cpu(0) float32
     * [  -inf, 0.9163]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray log();

    /**
     * Returns element-wise Base-2 logarithmic value of the input.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1000f, 1f, 150f});
     * jshell&gt; array.log10();
     * ND: (3) cpu(0) float32
     * [3.    , 0.    , 2.1761]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray log10();

    /**
     * Returns element-wise Base-2 logarithmic value of the input.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {8, 1f, 5f});
     * jshell&gt; array.log2();
     * ND: (3) cpu(0) float32
     * [3.    , 0.    , 2.3219]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray log2();

    /**
     * Computes the element-wise sine of the input array. The input should be in radians ( 2𝜋 rad
     * equals 360 degrees).
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 30f, 45f, 60f, 90f});
     * jshell&gt; array = array.mul(Math.PI).div(180f);
     * jshell&gt; array.sin();
     * ND: (5) cpu(0) float32
     * [0.    , 0.5   , 0.7071, 0.866 , 1.    ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray sin();

    /**
     * Computes the element-wise cosine of the input array. The input should be in radians ( 2𝜋 rad
     * equals 360 degrees).
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI/2, Math.PI});
     * jshell&gt; array.cos();
     * ND: (3) cpu(0) float64
     * [  1.0000000e+00,   6.1232340e-17,  -1.0000000e+00],
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray cos();

    /**
     * Computes the element-wise tangent of the input array. The input should be in radians ( 2𝜋
     * rad equals 360 degrees).
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {-Math.PI, Math.PI/2, Math.PI});
     * jshell&gt; array.tan();
     * ND: (3) cpu(0) float64
     * [  1.2246468e-16,   1.6331239e+16,  -1.2246468e-16],
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray tan();

    /**
     * Returns element-wise inverse sine of the input array. The input should be in the range [-1,
     * 1]. The output is in the closed interval of [ −𝜋/2 , 𝜋/2 ].
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, -1f, 0f});
     * jshell&gt; array.asin();
     * ND: (3) cpu(0) float64
     * [ 1.5708, -1.5708,  0.    ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray asin();

    /**
     * Returns element-wise inverse cosine of the input array. The input should be in the range [-1,
     * 1]. The output is in the closed interval of [ −𝜋/2 , 𝜋/2 ].
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, -1f});
     * jshell&gt; array.acos();
     * ND: (2) cpu(0) float64
     * [0.    , 3.1416]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray acos();

    /**
     * Returns element-wise inverse tangent of the input array. The input should be in the range
     * [-1, 1]. The output is in the closed interval of [ −𝜋/2 , 𝜋/2 ].
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f});
     * jshell&gt; array.atan();
     * ND: (2) cpu(0) float64
     * [0.    , 0.7854]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray atan();

    /**
     * Returns the hyperbolic sine of the input array, computed element-wise.
     * 𝑠𝑖𝑛ℎ(𝑥)=0.5×(𝑒𝑥𝑝(𝑥)−𝑒𝑥𝑝(−𝑥))
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.sinh();
     * ND: (2) cpu(0) float64
     * [ 0.    , 11.5487]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray sinh();

    /**
     * Returns the hyperbolic cosine of the input array, computed element-wise.
     * 𝑐𝑜𝑠ℎ(𝑥)=0.5×(𝑒𝑥𝑝(𝑥)+𝑒𝑥𝑝(−𝑥))
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.cosh();
     * ND: (2) cpu(0) float64
     * [ 1.    , 11.592 ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray cosh();

    /**
     * Returns the hyperbolic tangent of the input array, computed element-wise.
     * 𝑡𝑎𝑛ℎ(𝑥)=𝑠𝑖𝑛ℎ(𝑥)/𝑐𝑜𝑠ℎ(𝑥)
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.tanh();
     * ND: (2) cpu(0) float64
     * [0.    , 0.9963]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray tanh();

    /**
     * Returns the element-wise inverse hyperbolic sine of the input array, computed element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {Math.E, 10});
     * jshell&gt; array.asinh();
     * ND: (2) cpu(0) float64
     * [1.7254, 2.9982]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray asinh();

    /**
     * Returns the element-wise inverse hyperbolic cosine of the input array, computed element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {Math.E, 10});
     * jshell&gt; array.acosh();
     * ND: (2) cpu(0) float64
     * [1.6575, 2.9932]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray acosh();

    /**
     * Returns the element-wise inverse hyperbolic tangent of the input array, computed
     * element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, -0.5});
     * jshell&gt; array.atanh();
     * ND: (2) cpu(0) float64
     * [ 0.    , -0.5493]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray atanh();

    /**
     * Converts each element of the input array from radians to degrees.
     * 𝑑𝑒𝑔𝑟𝑒𝑒𝑠([0,𝜋/2,𝜋,3𝜋/2,2𝜋])=[0,90,180,270,360].
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6).mul(Math.PI / 3);
     * jshell&gt; array.toDegrees();
     * ND: (6) cpu(0) float32
     * [  0.,  60., 120., 180., 240., 300.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray toDegrees();

    /**
     * Converts each element of the input array from degrees to radians.
     * 𝑟𝑎𝑑𝑖𝑎𝑛𝑠([0,90,180,270,360])=[0,𝜋/2,𝜋,3𝜋/2,2𝜋]
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6).mul(60);
     * jshell&gt; array.toRadians();
     * ND: (6) cpu(0) float32
     * [0.    , 1.0472, 2.0944, 3.1416, 4.1888, 5.236 ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray toRadians();

    ////////////////////////////////////////
    // Operators: Reduction
    ////////////////////////////////////////

    /**
     * Returns the maximum of an {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4).reshape(2,2);
     * jshell&gt; array
     * ND: (2, 2) cpu(0) float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.max() // Maximum of the flattened array
     * ND: () cpu(0) float32
     * 3.
     * </pre>
     *
     * @return the max of the {@code NDArray}
     */
    NDArray max();

    /**
     * Finds the max over the given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4).reshape(2,2);
     * jshell&gt; array
     * ND: (2, 2) cpu(0) float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{0}) // Maximum along the first axis
     * ND: (2) cpu(0) float32
     * [2., 3.]
     * jshell&gt; array.max(new int[]{1}) // Maximum along the first axis
     * ND: (2) cpu(0) float32
     * [1., 3.]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @return an NDArray with the specified axes removed from the Shape containing the max
     * @see NDArray#max(int[], boolean)
     */
    default NDArray max(int[] axes) {
        return max(axes, false);
    }

    /**
     * Finds the max over the given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4).reshape(2,2);
     * jshell&gt; array
     * ND: (2, 2) cpu(0) float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{0}, true) // Maximum along the first axis and keep dimension
     * ND: (1, 2) cpu(0) float32
     * [[2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{1}, true) // Maximum along the first axis and keep dimension
     * ND: (2, 1) cpu(0) float32
     * [[1.],
     *  [3.],
     * ]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array.
     * @return an NDArray after the max
     */
    NDArray max(int[] axes, boolean keepDims);

    /**
     * Finds the min of all elements in the {@code NDArray}.
     *
     * @return the min
     */
    NDArray min();

    /**
     * Finds the min over the given axes.
     *
     * @param axes the axes to find the min over
     * @return an NDArray with the specified axes removed from the Shape containing the min
     * @see NDArray#min(int[], boolean)
     */
    default NDArray min(int[] axes) {
        return min(axes, false);
    }

    /**
     * Finds the min over the given axes.
     *
     * @param axes the axes to find the min over
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return an NDArray after the min
     */
    NDArray min(int[] axes, boolean keepDims);

    /**
     * Sums all elements in the {@code NDArray}.
     *
     * @return the sum
     */
    NDArray sum();

    /**
     * Sums over the given axes.
     *
     * @param axes the axes to sum over
     * @return an NDArray with the specified axes removed from the Shape containing the sum
     * @see NDArray#sum(int[], boolean)
     */
    default NDArray sum(int[] axes) {
        return sum(axes, false);
    }

    /**
     * Sums over the given axes.
     *
     * @param axes the axes to sum over
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return an NDArray after the sum
     */
    NDArray sum(int[] axes, boolean keepDims);

    /**
     * Finds the product of all elements in the {@code NDArray}.
     *
     * @return the product of all elements
     */
    NDArray prod();

    /**
     * Finds the product over the given axes.
     *
     * @param axes the axes to prod over
     * @return an NDArray with the specified axes removed from the Shape containing the prod
     * @see NDArray#prod(int[], boolean)
     */
    default NDArray prod(int[] axes) {
        return prod(axes, false);
    }

    /**
     * Finds the product over the given axes.
     *
     * @param axes the axes to prod over
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return an NDArray after the prod.
     */
    NDArray prod(int[] axes, boolean keepDims);

    /**
     * Finds the mean of all elements in the {@code NDArray}.
     *
     * @return the mean
     */
    NDArray mean();

    /**
     * Finds the mean over the given axes.
     *
     * @param axes the axes to find the mean over
     * @return an NDArray with the specified axes removed from the Shape containing the mean
     * @see NDArray#mean(int[], boolean)
     */
    default NDArray mean(int[] axes) {
        return mean(axes, false);
    }

    /**
     * Finds the mean over the given axes.
     *
     * @param axes the axes to find the mean over
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return an NDArray after the mean
     */
    NDArray mean(int[] axes, boolean keepDims);

    /**
     * Returns the sum along diagonals of the {@link NDArray}.
     *
     * <p>If {@link NDArray} is 2-D, the sum along its diagonal is returned. If {@link NDArray} has
     * more than two dimensions, then the axes specified by axis1 and axis2 are used to determine
     * the 2-D sub-arrays whose traces are returned. The shape of the resulting {@link NDArray} is
     * the same as that of a with axis1 and axis2 removed.
     *
     * @return If a is 2-D, the sum along the diagonal is returned. If {@link NDArray} has larger
     *     dimensions, then a {@link NDArray} of sums along diagonals is returned.
     */
    default NDArray trace() {
        return trace(0, 0, 1);
    }

    /**
     * Returns the sum along diagonals of the array.
     *
     * <p>If a is 2-D, the sum along its diagonal with the given offset is returned, i.e., the sum
     * of elements a[i,i+offset] for all i. If a has more than two dimensions, then the axes
     * specified by axis1 and axis2 are used to determine the 2-D sub-arrays whose traces are
     * returned. The shape of the resulting array is the same as that of a with axis1 and axis2
     * removed.
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @return If a is 2-D, the sum along the diagonal is returned. If a has larger dimensions, then
     *     an array of sums along diagonals is returned.
     */
    default NDArray trace(int offset) {
        return trace(offset, 0, 1);
    }

    /**
     * Returns the sum along diagonals of the array.
     *
     * <p>If a is 2-D, the sum along its diagonal with the given offset is returned, i.e., the sum
     * of elements a[i,i+offset] for all i. If a has more than two dimensions, then the axes
     * specified by axis1 and axis2 are used to determine the 2-D sub-arrays whose traces are
     * returned. The shape of the resulting array is the same as that of a with axis1 and axis2
     * removed.
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @param axis1 axes to be used as the first axis of the 2-D sub-arrays from which the diagonals
     *     should be taken
     * @param axis2 axes to be used as the second axis of the 2-D sub-arrays from which the
     *     diagonals should be taken
     * @return If a is 2-D, the sum along the diagonal is returned. If a has larger dimensions, then
     *     an array of sums along diagonals is returned.
     */
    NDArray trace(int offset, int axis1, int axis2);

    ////////////////////////////////////////
    // Operators: Shapes and Arrays Manipulation
    ////////////////////////////////////////

    /**
     * Splits the array into given sections of new NDArrays along the given axis.
     *
     * @param sections the array will be divided into N (sections) equal arrays along axis
     * @return an NDList with size(axis) NDArrays with shape {@code this.shape.remove(axis) }
     * @see NDArray#split(int, int)
     */
    default NDList split(int sections) {
        return split(sections, 0);
    }

    /**
     * Splits the array into given indices of new NDArrays along the given axis.
     *
     * @param indices the entries indicate where along axis the array is split. If an index exceeds
     *     the dimension of the array along axis, an empty sub-array is returned correspondingly.
     * @return an NDList with size(axis) NDArrays with shape {@code this.shape.remove(axis) }
     * @see NDArray#split(int[], int)
     */
    default NDList split(int[] indices) {
        return split(indices, 0);
    }

    /**
     * Splits the array into a given number of sections of new NDArrays along the given axis.
     *
     * @param sections the array will be divided into N (sections) equal arrays along axis
     * @param axis the axis to split along
     * @return an NDList with numOutputs NDArrays with shape {@code (this.shape.axis /= axis) }
     * @throws IllegalArgumentException thrown if the numOutputs does not equally divide the given
     *     axis
     */
    default NDList split(int sections, int axis) {
        long axisSize = getShape().getShape()[axis];
        if (axisSize % sections != 0) {
            throw new IllegalArgumentException("array split does not result in an equal division");
        }
        long sectionSize = axisSize / sections;
        int[] indices = IntStream.range(0, sections).map(i -> (int) (i * sectionSize)).toArray();
        return split(indices, axis);
    }

    /**
     * Splits the array into given indices of new NDArrays along the given axis.
     *
     * @param indices the entries indicate where along axis the array is split. If an index exceeds
     *     the dimension of the array along axis, an empty sub-array is returned correspondingly.
     * @param axis the axis to split along
     * @return an NDList with numOutputs NDArrays with shape {@code (this.shape.axis /= axis) }
     */
    NDList split(int[] indices, int axis);

    /**
     * Flattens the array into a 1D NDArray in row-major order.
     *
     * <p>To flatten in column-major order, first transpose the NDArray
     *
     * @return a 1D NDArray of equal size
     */
    NDArray flatten();

    /**
     * Reshapes the NDArray to the given shape.
     *
     * @param newShape the long array to reshape into. Must have equal size to the current shape.
     * @return a reshaped NDArray
     * @throws IllegalArgumentException Thrown if the given shape does not match the size of the
     *     current shape
     */
    default NDArray reshape(long... newShape) {
        return reshape(new Shape(newShape));
    }

    /**
     * Reshapes the NDArray to the given shape.
     *
     * <p>You can reshape it to match another NDArray by calling {@code a.reshape(b.getShape()) }
     *
     * @param shape the shape to reshape into. Must have equal size to the current shape.
     * @return a reshaped NDArray
     * @throws IllegalArgumentException Thrown if the given shape does not match the size of the
     *     current shape
     */
    NDArray reshape(Shape shape);

    /**
     * Expands the shape of a {@code NDArray}.
     *
     * <p>Inserts a new axis that will appear at the axis position in the expanded {@code NDArray}.
     *
     * @param axis the position in the expanded axes where the new axis is placed
     * @return the output array. The number of dimensions is one greater than that of the input
     *     array.
     */
    NDArray expandDims(int axis);

    /**
     * Removes all singleton dimensions from the NDArray shape.
     *
     * @return an output array of same size and data without singleton dimensions
     */
    default NDArray squeeze() {
        long[] shape = getShape().getShape();
        return squeeze(IntStream.range(0, shape.length).filter(i -> shape[i] == 1).toArray());
    }

    /**
     * Removes a singleton dimension at the given axis.
     *
     * @param axis the axis at which to remove the singleton dimension
     * @return an output array of same size and data without the axis at part of the shape
     * @throws IllegalArgumentException Thrown if the given axis is not a singleton dimension
     */
    default NDArray squeeze(int axis) {
        return squeeze(new int[] {axis});
    }

    /**
     * Removes singleton dimensions at the given axes.
     *
     * @param axes the axes at which to remove the singleton dimensions
     * @return an output array of same size and data without the axes at part of the shape
     * @throws IllegalArgumentException Thrown if any of the given axes are not a singleton
     *     dimension
     */
    NDArray squeeze(int[] axes);

    /**
     * Joins a sequence of {@code NDArray}s in NDList along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last
     * dimension.
     *
     * @param arrays the input NDList. Each {@code NDArray} must have the same shape.
     * @param axis the axis in the result array along which the input arrays are stacked
     * @return the {@code NDArray}. The stacked array has one more dimension than the input arrays.
     */
    NDArray stack(NDList arrays, int axis);

    /**
     * Joins a sequence of {@code NDArray}s in NDList along axis 0.
     *
     * @param arrays the input NDList. each {@code NDArray} must have the same shape.
     * @return the {@code NDArray}. The stacked array has one more dimension than the input arrays.
     */
    default NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a {@code NDArray} along a new axis.
     *
     * @param array the input {@code NDArray}. It must have the same shape as the {@code NDArray}
     *     that invokes the function.
     * @param axis the axis in the result array along which the input arrays are stacked
     * @return the {@code NDArray}. The stacked array has one more dimension than the input arrays.
     */
    default NDArray stack(NDArray array, int axis) {
        return stack(new NDList(array), axis);
    }

    /**
     * Joins a {@code NDArray} along axis 0.
     *
     * @param array the input {@code NDArray}. It must have the same shape with {@code NDArray} that
     *     invoke the function.
     * @return the {@code NDArray}. The stacked array has one more dimension than the input arrays.
     */
    default NDArray stack(NDArray array) {
        return stack(new NDList(array));
    }

    /**
     * Joins a NDList along an existing axis.
     *
     * @param arrays an NDList with input {@code NDArray} of the same shape, except in the dimension
     *     corresponding to `axis` (the first, by default)
     * @param axis the axis along which the arrays will be joined
     * @return the concatenated {@code NDArray}
     */
    NDArray concat(NDList arrays, int axis);

    /**
     * Joins a NDList along axis 0.
     *
     * @param arrays an NDList with input {@code NDArray} of the same shape, except in the dimension
     *     corresponding to `axis` (the first, by default)
     * @return the concatenated {@code NDArray}
     */
    default NDArray concat(NDList arrays) {
        return concat(arrays, 0);
    }

    /**
     * Joins a {@code NDArray} along an existing axis.
     *
     * @param array an {@code NDArray} of the same shape, except in the dimension corresponding to
     *     `axis` (the first, by default)
     * @param axis the axis along which the arrays will be joined
     * @return the concatenated {@code NDArray}
     */
    default NDArray concat(NDArray array, int axis) {
        return concat(new NDList(array), axis);
    }

    /**
     * Joins a {@code NDArray} along axis 0.
     *
     * @param array an {@code NDArray} of the same shape, except in the dimension corresponding to
     *     `axis` (the first, by default)
     * @return the concatenated {@code NDArray}
     */
    default NDArray concat(NDArray array) {
        return concat(new NDList(array));
    }

    ////////////////////////////////////////
    // Operators: Logical Op
    ////////////////////////////////////////

    /**
     * Computes the truth value of this {@code NDArray} AND other {@code NDArray} element-wise.
     *
     * @param other the second NDArray to operate on
     * @return the boolean result of the logical OR operation applied to the elements of this {@code
     *     NDArray}
     */
    NDArray logicalAnd(NDArray other);

    /**
     * Computes the truth value of this {@code NDArray} OR other {@code NDArray} element-wise.
     *
     * @param other the second NDArray to operate on
     * @return the boolean result of the logical OR operation applied to the elements of this {@code
     *     NDArray}
     */
    NDArray logicalOr(NDArray other);

    /**
     * Computes the truth value of this {@code NDArray} XOR other {@code NDArray} element-wise..
     *
     * @param other the second NDArray to operate on
     * @return the boolean result of the logical XOR operation applied to the elements of this
     *     {@code NDArray}
     */
    NDArray logicalXor(NDArray other);

    /**
     * Computes the truth value of NOT this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    NDArray logicalNot();

    ////////////////////////////////////////
    // Operators: Other
    ////////////////////////////////////////

    /**
     * Performs an indirect sort of the NDArray ascending on the last dimension.
     *
     * @return an array of indices corresponding to elements in the NDArray on the axis, the output
     *     DataType is always {@link DataType#INT32}
     * @see NDArray#argsort(int, boolean)
     */
    default NDArray argsort() {
        return argsort(-1, true);
    }

    /**
     * Performs an indirect sort of the NDArray ascending on the given dimension.
     *
     * @param axis the axis to sort along
     * @return an array of indices corresponding to elements in the NDArray on the axis, the output
     *     DataType is always {@link DataType#INT32}
     * @see NDArray#argsort(int, boolean)
     */
    default NDArray argsort(int axis) {
        return argsort(axis, true);
    }

    /**
     * Performs an indirect sort of the NDArray on the given dimension.
     *
     * @param axis the axis to sort along
     * @param ascending whether to sort ascending
     * @return an array of indices corresponding to elements in the NDArray on the axis, the output
     *     DataType is always {@link DataType#INT32}
     */
    NDArray argsort(int axis, boolean ascending);

    /**
     * Returns a sorted copy of an input array along the given axis.
     *
     * @param axis the axis along which to sort
     * @return the sorted NDArray
     */
    NDArray sort(int axis);

    /**
     * Returns a sorted copy of an flattened input array.
     *
     * @return the sorted NDArray
     */
    NDArray sort();

    /**
     * Returns the softmax on the specified axis.
     *
     * @param axis the axis to sort along, -1 for the last axis
     * @return the softmax
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int[], double)
     */
    default NDArray softmax(int axis) {
        return softmax(new int[] {axis}, 1);
    }

    /**
     * Returns the softmax on the specified axis.
     *
     * @param axis the axis to sort along, -1 for the last axis
     * @param temperature the exponent multiplier Beta in the softmax
     * @return the softmax
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int[], double)
     */
    default NDArray softmax(int axis, double temperature) {
        return softmax(new int[] {axis}, temperature);
    }

    /**
     * Returns the softmax across the specified axes.
     *
     * @param axes the axes to compute the softmax of. An empty array indicates computing the
     *     softmax for the whole array.
     * @return the softmax
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     */
    default NDArray softmax(int[] axes) {
        return softmax(axes, 1);
    }

    /**
     * Returns the softmax across the specified axes.
     *
     * @param axes the axes to compute the softmax of. An empty array indicates computing the
     *     softmax for the whole array.
     * @param temperature the exponent multiplier Beta in the softmax
     * @return the softmax
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int[], double)
     */
    NDArray softmax(int[] axes, double temperature);

    /**
     * Returns the cumulative sum along an axis in-place.
     *
     * @param axis the axis along which the cumulative sum is computed
     * @return this object
     */
    NDArray cumsumi(int axis);

    /**
     * Returns the cumulative sum over the flattened array in-place.
     *
     * @return this object
     */
    NDArray cumsumi();

    /**
     * Returns the cumulative sum along an axis.
     *
     * @param axis the axis along which the cumulative sum is computed
     * @return the cumulative sum along the specified dimension
     */
    NDArray cumsum(int axis);

    /**
     * Returns the cumulative sum over the flattened array.
     *
     * @return the cumulative sum along the specified dimension
     */
    NDArray cumsum();

    /**
     * Returns the binary NDArray with value {@code true} where this array's entries are infinite,
     * or {@code false} where they are not infinite.
     *
     * @return the binary array with value {@code true} if the array's entries are infinite
     */
    NDArray isInfinite();

    /**
     * Returns the binary NDArray with value {@code true} where this array's entries are NaN, or
     * {@code false} where they are not NaN.
     *
     * @return the binary array with value {@code true} if the array's entries are NaN
     */
    NDArray isNaN();

    /**
     * Returns a mask on whether each element matches the given index.
     *
     * @param index the index of values to set to true
     * @return a new boolean NDArray where values are {@code true} if it matches the index
     */
    NDArray createMask(NDIndex index);

    /**
     * Returns a mask on whether each element matches the given condition.
     *
     * @param predicate a predicate to apply to each element of the array
     * @return a new boolean NDArray where values are {@code true} if it matches the predicate
     */
    NDArray createMask(Predicate<Number> predicate);

    /**
     * Repeats the array in tiles a given number of times.
     *
     * @param repeats the number of times to repeat for each dimension
     * @return a NDArray that has been tiled
     */
    NDArray tile(long repeats);

    /**
     * Repeats the array in tiles a given number of times along the given axis.
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each dimension
     * @return an NDArray that has been tiled
     * @throws IllegalArgumentException Thrown for invalid axis
     */
    NDArray tile(int axis, long repeats);

    /**
     * Repeats the array in tiles a given number of times.
     *
     * @param repeats the number of times to repeat along each axis
     * @return an NDArray that has been tiled
     */
    NDArray tile(long[] repeats);

    /**
     * Repeats the array in tiles a given number of times to match the desired shape.
     *
     * <p>If the desired shape has fewer dimensions that the array, it will tile against the final
     * dimensions.
     *
     * @param desiredShape the shape that should be converted to
     * @return an NDArray that has been tiled
     */
    NDArray tile(Shape desiredShape);

    /**
     * Returns a sparse representation of {@code NDArray}.
     *
     * @param fmt the {@code SparseFormat} of the NDArray
     * @return the result {@code NDArray} NDArray
     */
    NDArray toSparse(SparseFormat fmt);

    /**
     * Repeats each array element a given number of times.
     *
     * @param repeats the number of times to repeat for each dimension
     * @return an NDArray that has been tiled
     */
    NDArray repeat(long repeats);

    /**
     * Repeats each array element a given number of times along the given axis.
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each dimension
     * @return an NDArray that has been tiled
     * @throws IllegalArgumentException Thrown for invalid axis
     */
    NDArray repeat(int axis, long repeats);

    /**
     * Repeats each array element a given number of times for each axis.
     *
     * @param repeats the number of times to repeat along each axis
     * @return an NDArray that has been tiled
     */
    NDArray repeat(long[] repeats);

    /**
     * Repeats each array element to match the desired shape.
     *
     * <p>If the desired shape has fewer dimensions that the array, it will tile against the final
     * dimensions.
     *
     * @param desiredShape the shape that should be converted to
     * @return an NDArray that has been tiled
     */
    NDArray repeat(Shape desiredShape);

    /**
     * Performs a copy matrix multiplication.
     *
     * @param other the other matrix to perform matrix multiplication with
     * @return the result of the matrix multiplication
     */
    NDArray dot(NDArray other);

    /**
     * Clips (limit) the values in an array.
     *
     * <p>Given an interval, values outside the interval are clipped to the interval edges. For
     * example, if an interval of ``[0, 1]`` is specified, values smaller than 0 become 0, and
     * values larger than 1 become 1.
     *
     * @param min the minimum value double type
     * @param max the maximum value double type
     * @return an {@code NDArray} with the elements of `a`, but where values &lt; `min` are replaced
     *     with `min`, and those &gt; `max` with `max`
     */
    NDArray clip(double min, double max);

    /**
     * Clips (limit) the values in an array.
     *
     * <p>Given an interval, values outside the interval are clipped to the interval edges. For
     * example, if an interval of ``[0, 1]`` is specified, values smaller than 0 become 0, and
     * values larger than 1 become 1.
     *
     * @param min the minimum value int type
     * @param max the maximum value int type
     * @return an {@code NDArray} with the elements of `a`, but where values &lt; `min` are replaced
     *     with `min`, and those &gt; `max` with `max`
     */
    default NDArray clip(int min, int max) {
        return clip((double) min, (double) max);
    }

    /**
     * Interchanges two axes of an array.
     *
     * @param axis1 the first axis
     * @param axis2 the second axis
     * @return the swapped axes view
     */
    default NDArray swapAxes(int axis1, int axis2) {
        int[] dims = IntStream.range(0, getShape().dimension()).toArray();
        int tmp = dims[axis1];
        dims[axis1] = dims[axis2];
        dims[axis2] = tmp;
        return transpose(dims);
    }

    /**
     * Reverses the order of the dimensions in the {@code NDArray}.
     *
     * @return the newly permuted array
     */
    NDArray transpose();

    /**
     * Reorders the dimensions in the {@code NDArray}.
     *
     * @param axes the axes to swap to
     * @return the newly permuted array
     * @throws IllegalArgumentException thrown when passing a axis that is greater than the actual
     *     number of dimensions
     */
    NDArray transpose(int... axes);

    /**
     * Broadcasts this NDArray to be the specified shape.
     *
     * @param shape the new shape of this NDArray
     * @return the broadcasted NDArray
     */
    NDArray broadcast(Shape shape);

    /**
     * Broadcasts this NDArray to be the specified shape.
     *
     * @param shape the new shape of this NDArray
     * @return the broadcasted NDArray
     */
    default NDArray broadcast(long... shape) {
        return broadcast(new Shape(shape));
    }

    /**
     * Checks 2 NDArrays for equal shapes.
     *
     * <pre>
     * Shapes are considered equal if:
     * (a) Both arrays have equal rank, and
     * (b) size(0)...size(rank()-1) are equal for both arrays
     * </pre>
     *
     * @param other the other NDArray
     * @return {@code true} if the shapes are the same
     */
    default boolean equalShapes(NDArray other) {
        return getShape().equals(other.getShape());
    }

    /**
     * Returns the index of the highest value.
     *
     * @return an array containing indices
     */
    NDArray argmax();

    /**
     * Returns the index of the highest value along specified axi(e)s.
     *
     * @param axis the axis along which to find argmax
     * @return an array containing indices
     */
    NDArray argmax(int axis);

    /**
     * Returns the index of the lowest value.
     *
     * @return an array containing indices
     */
    NDArray argmin();

    /**
     * Returns the index of the lowest value along specified axi(e)s.
     *
     * @param axis the axis along which to find argmax
     * @return an array containing indices
     */
    NDArray argmin(int axis);

    /**
     * Returns percentile value for this {@code NDArray}.
     *
     * @param percentile the target percentile in range of 0..100
     * @return the result {@code NDArray} NDArray
     */
    Number percentileNumber(Number percentile);

    /**
     * Returns median value for this {@code NDArray}.
     *
     * @return the median value for array
     */
    Number medianNumber();

    /**
     * Returns median along given dimension(s).
     *
     * @param axes the axes along which to perform the median operation
     * @return the median along the specified axes
     */
    NDArray median(int... axes);

    /**
     * Returns median along given dimension(s).
     *
     * @param percentile the target percentile in range of 0..100
     * @param axes the dimension to calculate percentile for
     * @return the result {@code NDArray} NDArray
     */
    NDArray percentile(Number percentile, int... axes);

    // ------------ Sparse methods ------------

    /**
     * Returns a dense representation of the sparse {@code NDArray}.
     *
     * @return the result {@code NDArray} NDArray
     */
    NDArray toDense();

    /**
     * Returns the indices of elements that are non-zero.
     *
     * <p>Note that the behavior is slightly different from numpy.nonzero. Numpy returns a tuple of
     * NDArray, one for each dimension of NDArray. DJL nonzero returns only one NDArray with last
     * dimension containing all dimension of indices
     *
     * @return the indices of the elements that are non-zero
     */
    NDArray nonzero();

    /**
     * Returns {@code true} if this NDArray is special case: no-value {@code NDArray}.
     *
     * @return {@code true} if this NDArray is empty
     */
    boolean isEmpty();

    /**
     * Returns {@code true} if all elements within this array are non-zero or {@code true}.
     *
     * @return {@code true} if all elements within this array are non-zero or {@code true}
     */
    default boolean all() {
        // result of sum operator is int64 now
        return asType(DataType.BOOLEAN, false).sum().getLong() == size();
    }

    /**
     * Returns {@code true} if any of the elements within this array are non-zero or {@code true}.
     *
     * @return {@code true} if any of the elements within this array are non-zero or {@code true}
     */
    default boolean any() {
        return asType(DataType.BOOLEAN, false).sum().getLong() > 0;
    }

    /**
     * Returns {@code true} if none of the elements within this array are non-zero or {@code true}.
     *
     * @return {@code true} if none of the elements within this array are non-zero or {@code true}
     */
    default boolean none() {
        return asType(DataType.BOOLEAN, false).sum().getLong() == 0;
    }

    /**
     * Counts the number of non-zero values in the {@code NDArray}.
     *
     * @return the number of non-zero values in the {@code NDArray}
     */
    default long countNonzero() {
        return asType(DataType.BOOLEAN, false).sum().getLong();
    }

    /**
     * Counts the number of non-zero values in the {@code NDArray} along a given axis.
     *
     * @param axis the axis to operate on
     * @return the number of non-zero values in the {@code NDArray} along a given axis
     */
    default long countNonzero(int axis) {
        return asType(DataType.BOOLEAN, false).sum(new int[] {axis}).getLong();
    }

    /**
     * Returns an internal representative of Native {@code NDArray}.
     *
     * <p>This method should only be used by Engine provider
     *
     * @return an internal representative of Native NDArray
     */
    NDArrayEx getNDArrayInternal();

    /** {@inheritDoc} */
    @Override
    void close();
}
