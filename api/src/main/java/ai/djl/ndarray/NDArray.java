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
     * Converts the {@code NDArray} to a double array.
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
     * Converts the {@code NDArray} to a float array.
     *
     * @return a float array
     * @throws IllegalStateException when {@link DataType} of the {@code NDArray} mismatches
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
     * Converts the {@code NDArray} to an int array.
     *
     * @return an int array
     * @throws IllegalStateException when {@link DataType} of the {@code NDArray}} mismatches
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
     * Converts the {@code NDArray} to a long array.
     *
     * @return a long array
     * @throws IllegalStateException when {@link DataType} of the {@code NDArray} mismatches
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
     * Converts the {@code NDArray} to a byte array.
     *
     * @return a byte array
     * @throws IllegalStateException when {@link DataType} of the {@code NDArray} mismatches
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
     * Converts the {@code NDArray} to a uint8 array.
     *
     * @return a uint8 array
     * @throws IllegalStateException when {@link DataType} of the {@code NDArray} mismatches
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
     * Converts the {@code NDArray} to a boolean array.
     *
     * @return a boolean array
     * @throws IllegalStateException when {@link DataType} of the {@code NDArray} mismatches
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
     * Converts the {@code NDArray} to a Number array based on its {@link DataType}.
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
     * Converts the {@code NDArray} to a ByteBuffer.
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
     * Creates a new {@code NDArray} whose content is a copy of the {@code NDArray}'s content.
     *
     * @return the new {@code NDArray}
     */
    default NDArray slice() {
        // TODO: MXNet doesn't support view, return a copy for now
        return duplicate();
    }

    /**
     * Returns a copy of the {@code NDArray}.
     *
     * @return a copy of the {@code NDArray}
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
    default boolean shapeEquals(NDArray other) {
        return getShape().equals(other.getShape());
    }

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
     * @param equalNan whether to compare NaN’s as equal. If true, NaN’s in the {@code NDArray} will
     *     be considered equal to NaN’s in the other {@code NDArray}
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

    /**
     * Returns elements chosen from the {@code NDArray} or the other {@code NDArray} depending on
     * condition.
     *
     * <p>Given three {@code NDArray}s, condition, this, and other, returns an {@code NDArray} with
     * the elements from this or other, depending on whether the elements from condition {@code
     * NDArray} are {@code true} or {@code false}. If condition has the same shape as this, each
     * element in the output {@link NDArray} is from this if the corresponding element in the
     * condition is true, and from other if false.
     *
     * <p>Note that all non-zero values are interpreted as true in condition {@code NDArray}.
     *
     * @param condition the condition {@code NDArray}
     * @param other the other {@code NDArray}
     * @return the result {@code NDArray}
     */
    NDArray where(NDArray condition, NDArray other);

    ////////////////////////////////////////
    // Operators: Element Arithmetic
    ////////////////////////////////////////

    /**
     * Adds a number to the {@code NDArray} element-wise.
     *
     * @param n the number to add
     * @return the result {@code NDArray}
     */
    NDArray add(Number n);

    /**
     * Adds other {@code NDArray}s to the {@code NDArray} element-wise.
     *
     * <p>The shapes of the {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * @param others the other {@code NDArray}s to add
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray add(NDArray... others);

    /**
     * Subtracts a number from the {@code NDArray} element-wise.
     *
     * @param n the number to subtract from
     * @return the result {@code NDArray}
     */
    NDArray sub(Number n);

    /**
     * Subtracts the other {@code NDArray} from the {@code NDArray} element-wise.
     *
     * <p>The shapes of the {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * @param other the other {@code NDArray} to subtract from
     * @return the result {@code NDArray}
     */
    NDArray sub(NDArray other);

    /**
     * Multiplies the {@code NDArray} by a number element-wise.
     *
     * @param n the number to multiply by
     * @return the result {@code NDArray}
     */
    NDArray mul(Number n);

    /**
     * Multiplies the {@code NDArray} by other {@code NDArray}s element-wise.
     *
     * <p>The shapes of the {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * @param others the other {@code NDArray}s to multiply by
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray mul(NDArray... others);

    /**
     * Divides the {@code NDArray} by a number element-wise.
     *
     * @param n the number to divide values by
     * @return the result {@code NDArray}
     */
    NDArray div(Number n);

    /**
     * Divides the {@code NDArray} by the other {@code NDArray} element-wise.
     *
     * <p>The shapes of the {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * @param other the other {@code NDArray} to divide by
     * @return the result {@code NDArray}
     */
    NDArray div(NDArray other);

    /**
     * Returns element-wise remainder of division.
     *
     * @param n the divisor number
     * @return the result {@code NDArray}
     */
    NDArray mod(Number n);

    /**
     * Returns element-wise remainder of division.
     *
     * <p>The shapes of the {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * @param other the divisor {@code NDArray}
     * @return the result {@code NDArray}
     */
    NDArray mod(NDArray other);

    /**
     * Takes the power of the {@code NDArray} with a number element-wise.
     *
     * @param n the number to take the power with
     * @return the result {@code NDArray}
     */
    NDArray pow(Number n);

    /**
     * Takes the power of the {@code NDArray} with the other {@code NDArray} element-wise.
     *
     * @param other the other {@code NDArray} to take the power with
     * @return the result {@code NDArray}
     */
    NDArray pow(NDArray other);

    /**
     * Adds a number to the {@code NDArray} element-wise in place.
     *
     * @param n the number to add
     * @return the result {@code NDArray}
     */
    NDArray addi(Number n);

    /**
     * Adds other {@code NDArray}s to the {@code NDArray} element-wise in place.
     *
     * <p>The shapes of the {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * @param others the other {@code NDArray}s to add
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray addi(NDArray... others);

    /**
     * Subtracts a number from the {@code NDArray} element-wise in place.
     *
     * @param n the number to subtract
     * @return the result {@code NDArray}
     */
    NDArray subi(Number n);

    /**
     * Subtracts the other {@code NDArray} from the {@code NDArray} element-wise in place.
     *
     * <p>The shapes of the {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * @param other the other {@code NDArray} to subtract from
     * @return the result {@code NDArray}
     */
    NDArray subi(NDArray other);

    /**
     * Multiplies the {@code NDArray} by a number element-wise in place.
     *
     * @param n the number to multiply by
     * @return the result {@code NDArray}
     */
    NDArray muli(Number n);

    /**
     * Multiplies the {@code NDArray} by other {@code NDArray}s element-wise in place.
     *
     * <p>The shapes of the {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * @param others the other NDArrays to multiply with
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray muli(NDArray... others);

    /**
     * Divides the {@code NDArray} by a number element-wise in place.
     *
     * @param n the number to divide values by
     * @return the array after applying division operation
     */
    NDArray divi(Number n);

    /**
     * Divides the {@code NDArray} by the other {@code NDArray} element-wise in place.
     *
     * <p>The shapes of the {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * @param other the other {@code NDArray} to divide by
     * @return the result of the divide
     */
    NDArray divi(NDArray other);

    /**
     * Returns element-wise remainder of division in place.
     *
     * @param n the divisor number
     * @return the result {@code NDArray}
     */
    NDArray modi(Number n);

    /**
     * Returns in place element-wise remainder of division in place.
     *
     * @param other the divisor {@code NDArray}
     * @return the result of the divide
     */
    NDArray modi(NDArray other);

    /**
     * Takes the power of the {@code NDArray} with a number element-wise in place.
     *
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    NDArray powi(Number n);

    /**
     * Takes the power of the {@code NDArray} with the other {@code NDArray} element-wise in place.
     *
     * @param other the other {@code NDArray} to take the power with
     * @return the result {@code NDArray}
     */
    NDArray powi(NDArray other);

    ////////////////////////////////////////
    // Operators: Basic Numeric
    ////////////////////////////////////////

    /**
     * Returns the numerical negative {@code NDArray} element-wise.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5);
     * jshell&gt; array.neg();
     * ND: (5) cpu(0) float32
     * [-0., -1., -2., -3., -4.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray neg();

    /**
     * Returns the numerical negative {@code NDArray} element-wise in place.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5);
     * jshell&gt; array.negi();
     * jshell&gt; array;
     * ND: (5) cpu(0) float32
     * [-0., -1., -2., -3., -4.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray negi();

    /**
     * Returns the absolute value of the {@code NDArray} element-wise.
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
     * Returns the square of the {@code NDArray} element-wise.
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
     * Returns the cube-root of the {@code NDArray} element-wise.
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
     * Returns the floor of the {@code NDArray} element-wise.
     *
     * <p>The floor of the scalar x is the largest integer i, such that i &lt;= x. It is often
     * denoted as \lfloor x \rfloor.
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
     * Returns the ceiling of the {@code NDArray} element-wise.
     *
     * <p>The ceil of the scalar x is the smallest integer i, such that i &gt;= x. It is often
     * denoted as \lceil x \rceil.
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
     * Returns the round of the {@code NDArray} element-wise.
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
     * Returns the truncated value of the {@code NDArray} element-wise.
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
     * Returns the exponential value of the {@code NDArray} element-wise.
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
     * Returns the natural logarithmic value of the {@code NDArray} element-wise.
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
     * Returns the base 10 logarithm of the {@code NDArray} element-wise.
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
     * Returns the base 2 logarithm of the {@code NDArray} element-wise.
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
     * Returns the trigonometric sine of the {@code NDArray} element-wise.
     *
     * <p>The input should be in radians (2𝜋 radians equals 360 degrees).
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
     * Returns the trigonometric cosine of the {@code NDArray} element-wise.
     *
     * <p>The input should be in radians (2𝜋 radians equals 360 degrees).
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
     * Returns the trigonometric tangent of the {@code NDArray} element-wise.
     *
     * <p>The input should be in radians (2𝜋 radians equals 360 degrees).
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
     * Returns the inverse trigonometric sine of the {@code NDArray} element-wise.
     *
     * <p>The input should be in the range [-1, 1]. The output is in the closed interval of [−𝜋/2,
     * 𝜋/2].
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
     * Returns the inverse trigonometric cosine of the {@code NDArray} element-wise.
     *
     * <p>The input should be in the range [-1, 1]. The output is in the closed interval of [−𝜋/2,
     * 𝜋/2].
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
     * Returns the inverse trigonometric tangent of the {@code NDArray} element-wise.
     *
     * <p>The input should be in the range [-1, 1]. The output is in the closed interval of [−𝜋/2,
     * 𝜋/2].
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
     * Returns the hyperbolic sine of the {@code NDArray} element-wise.
     *
     * <p>𝑠𝑖𝑛ℎ(𝑥)=0.5×(𝑒𝑥𝑝(𝑥)−𝑒𝑥𝑝(−𝑥))
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
     * Returns the hyperbolic cosine of the {@code NDArray} element-wise.
     *
     * <p>𝑐𝑜𝑠ℎ(𝑥)=0.5×(𝑒𝑥𝑝(𝑥)+𝑒𝑥𝑝(−𝑥))
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
     * Returns the hyperbolic tangent of the {@code NDArray} element-wise.
     *
     * <p>𝑡𝑎𝑛ℎ(𝑥)=𝑠𝑖𝑛ℎ(𝑥)/𝑐𝑜𝑠ℎ(𝑥)
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
     * Returns the inverse hyperbolic sine of the {@code NDArray} element-wise.
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
     * Returns the inverse hyperbolic cosine of the {@code NDArray} element-wise.
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
     * Returns the inverse hyperbolic tangent of the {@code NDArray} element-wise.
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
     * Converts the {@code NDArray} from radians to degrees element-wise.
     *
     * <p>𝑑𝑒𝑔𝑟𝑒𝑒𝑠([0, 𝜋/2, 𝜋, 3𝜋/2, 2𝜋]) = [0, 90, 180, 270, 360].
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
     * Converts the {@code NDArray} from degrees to radians element-wise.
     *
     * <p>𝑟𝑎𝑑𝑖𝑎𝑛𝑠([0, 90, 180, 270, 360])=[0, 𝜋/2, 𝜋, 3𝜋/2, 2𝜋]
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
     * Returns the maximum of the {@code NDArray}.
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
     * @return the maximum of the {@code NDArray}
     */
    NDArray max();

    /**
     * Returns the maximum of the {@code NDArray} along given axes.
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
     * @return the maximum of the {@code NDArray} with the specified axes removed from the Shape
     *     containing the max
     * @see NDArray#max(int[], boolean)
     */
    default NDArray max(int[] axes) {
        return max(axes, false);
    }

    /**
     * Returns the maximum of the {@code NDArray} along given axes.
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
     * @return the maximum of the {@code NDArray}
     */
    NDArray max(int[] axes, boolean keepDims);

    /**
     * Returns the minimum of the {@code NDArray}.
     *
     * @return the minimum of the {@code NDArray}
     */
    NDArray min();

    /**
     * Returns the minimum of the {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @return the minimum of the {@code NDArray} with the specified axes removed from the Shape
     *     containing the min
     * @see NDArray#min(int[], boolean)
     */
    default NDArray min(int[] axes) {
        return min(axes, false);
    }

    /**
     * Returns the minimum of the {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the minimum of the {@code NDArray}
     */
    NDArray min(int[] axes, boolean keepDims);

    /**
     * Returns the sum of the {@code NDArray}.
     *
     * @return the sum of the {@code NDArray}
     */
    NDArray sum();

    /**
     * Returns the minimum of the {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @return the sum of the {@code NDArray} with the specified axes removed from the Shape
     *     containing the sum
     * @see NDArray#sum(int[], boolean)
     */
    default NDArray sum(int[] axes) {
        return sum(axes, false);
    }

    /**
     * Returns the minimum of the {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the sum of the {@code NDArray}
     */
    NDArray sum(int[] axes, boolean keepDims);

    /**
     * Returns the product of the {@code NDArray}.
     *
     * @return the product of the {@code NDArray}
     */
    NDArray prod();

    /**
     * Returns the product of the {@code NDArray} elements over the given axes.
     *
     * @param axes the axes along which to operate
     * @return the product of the {@code NDArray} with the specified axes removed from the Shape
     *     containing the prod
     * @see NDArray#prod(int[], boolean)
     */
    default NDArray prod(int[] axes) {
        return prod(axes, false);
    }

    /**
     * Returns the product of the {@code NDArray} elements over the given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the product of the {@code NDArray}
     */
    NDArray prod(int[] axes, boolean keepDims);

    /**
     * Returns the average of the {@code NDArray}.
     *
     * @return the average of the {@code NDArray}
     */
    NDArray mean();

    /**
     * Returns the average of the {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @return the average of the {@code NDArray} with the specified axes removed from the Shape
     *     containing the mean
     * @see NDArray#mean(int[], boolean)
     */
    default NDArray mean(int[] axes) {
        return mean(axes, false);
    }

    /**
     * Returns the average of the {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the average of the {@code NDArray}
     */
    NDArray mean(int[] axes, boolean keepDims);

    /**
     * Returns the sum along diagonals of the {@link NDArray}.
     *
     * <p>If the {@link NDArray} is 2-D, the sum along its diagonal is returned. If the {@link
     * NDArray} has more than two dimensions, then the axes specified by axis1 and axis2 are used to
     * determine the 2-D sub-arrays whose traces are returned. The shape of the resulting {@link
     * NDArray} is the same as that of a with axis1 and axis2 removed.
     *
     * @return the sum along diagonals of the {@link NDArray}
     */
    default NDArray trace() {
        return trace(0, 0, 1);
    }

    /**
     * Returns the sum along diagonals of the {@link NDArray}.
     *
     * <p>If the {@link NDArray} is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If the {@link NDArray} has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The shape of the resulting array is the same as that of
     * the {@link NDArray} with axis1 and axis2 removed.
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @return the sum along diagonals of the {@link NDArray}
     */
    default NDArray trace(int offset) {
        return trace(offset, 0, 1);
    }

    /**
     * Returns the sum along diagonals of the {@link NDArray}.
     *
     * <p>If the {@link NDArray} is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If the {@link NDArray} has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The shape of the resulting array is the same as that of
     * the {@link NDArray} with axis1 and axis2 removed.
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @param axis1 axes to be used as the first axis of the 2-D sub-arrays from which the diagonals
     *     should be taken
     * @param axis2 axes to be used as the second axis of the 2-D sub-arrays from which the
     *     diagonals should be taken
     * @return the sum along diagonals of the {@link NDArray}
     */
    NDArray trace(int offset, int axis1, int axis2);

    ////////////////////////////////////////
    // Operators: Shapes and Arrays Manipulation
    ////////////////////////////////////////

    /**
     * Splits the {@link NDArray} into multiple sub{@link NDArray}s given sections along first axis.
     *
     * @param sections the {@link NDArray} will be divided into N (sections) equal {@link NDArray}
     * @return an {@link NDList} with size(axis) {@link NDArray}s with shape {@code
     *     this.shape.remove(axis) }
     * @see NDArray#split(int, int)
     */
    default NDList split(int sections) {
        return split(sections, 0);
    }

    /**
     * Splits the {@link NDArray} into multiple sub-{@link NDArray}s given indices along first axis.
     *
     * @param indices the entries indicate where along axis the {@link NDArray} is split. If an
     *     index exceeds the dimension of the {@link NDArray} along axis, an empty sub-{@link
     *     NDArray} is returned correspondingly.
     * @return an NDList with size(axis) {@link NDArray}s with shape {@code this.shape.remove(axis)
     *     }
     * @see NDArray#split(int[], int)
     */
    default NDList split(int[] indices) {
        return split(indices, 0);
    }

    /**
     * Splits the {@link NDArray} into multiple sub{@link NDArray}s given sections along the given
     * axis.
     *
     * @param sections the {@link NDArray} will be divided into N (sections) equal arrays along axis
     * @param axis the axis to split along
     * @return an {@link NDList} with numOutputs {@link NDArray}s with shape {@code (this.shape.axis
     *     /= axis) }
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
     * Splits the {@link NDArray} into multiple sub-{@link NDArray}s given indices along given axis.
     *
     * @param indices the entries indicate where along axis the {@link NDArray} is split. If an
     *     index exceeds the dimension of the {@link NDArray} along axis, an empty sub-array is
     *     returned correspondingly
     * @param axis the axis to split along
     * @return an {@link NDList} with numOutputs {@link NDArray}s with shape {@code (this.shape.axis
     *     /= axis) }
     */
    NDList split(int[] indices, int axis);

    /**
     * Flattens the {@link NDArray} into a 1-D {@link NDArray} in row-major order.
     *
     * <p>To flatten in column-major order, first transpose the {@link NDArray}
     *
     * @return a 1-D {@link NDArray} of equal size
     */
    NDArray flatten();

    /**
     * Reshapes the {@link NDArray} to the given shape.
     *
     * @param newShape the long array to reshape into. Must have equal size to the current shape
     * @return a reshaped {@link NDArray}
     * @throws IllegalArgumentException thrown if the given shape does not match the size of the
     *     current shape
     */
    default NDArray reshape(long... newShape) {
        return reshape(new Shape(newShape));
    }

    /**
     * Reshapes the {@link NDArray} to the given shape.
     *
     * <p>You can reshape it to match another NDArray by calling {@code a.reshape(b.getShape()) }
     *
     * @param shape the shape to reshape into. Must have equal size to the current shape
     * @return a reshaped {@link NDArray}
     * @throws IllegalArgumentException thrown if the given shape does not match the size of the
     *     current shape
     */
    NDArray reshape(Shape shape);

    /**
     * Expands the shape of a {@link NDArray}.
     *
     * <p>Inserts a new axis that will appear at the axis position in the expanded {@code NDArray}
     * shape.
     *
     * @param axis the position in the expanded axes where the new axis is placed
     * @return the result {@link NDArray}. The number of dimensions is one greater than that of the
     *     {@link NDArray}
     */
    NDArray expandDims(int axis);

    /**
     * Removes all singleton dimensions from the {@link NDArray} shape.
     *
     * @return a result {@link NDArray} of same size and data without singleton dimensions
     */
    default NDArray squeeze() {
        long[] shape = getShape().getShape();
        return squeeze(IntStream.range(0, shape.length).filter(i -> shape[i] == 1).toArray());
    }

    /**
     * Removes a singleton dimension at the given axis.
     *
     * @param axis the axis at which to remove the singleton dimension
     * @return a result {@link NDArray} of same size and data without the axis at part of the shape
     * @throws IllegalArgumentException thrown if the given axis is not a singleton dimension
     */
    default NDArray squeeze(int axis) {
        return squeeze(new int[] {axis});
    }

    /**
     * Removes singleton dimensions at the given axes.
     *
     * @param axes the axes at which to remove the singleton dimensions
     * @return a result {@link NDArray} of same size and data without the axes at part of the shape
     * @throws IllegalArgumentException thrown if any of the given axes are not a singleton
     *     dimension
     */
    NDArray squeeze(int[] axes);

    /**
     * Joins a sequence of {@link NDArray}s in {@link NDList} along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last
     * dimension.
     *
     * @param arrays the input {@link NDList}. Each {@link NDArray} in the {@link NDList} must have
     *     the same shape as the {@link NDArray}
     * @param axis the axis in the result {@code NDArray} along which the input {@link NDList} are
     *     stacked
     * @return the result {@link NDArray}. The stacked {@code NDArray} has one more dimension than
     *     the the {@link NDArray}
     */
    NDArray stack(NDList arrays, int axis);

    /**
     * Joins a sequence of {@link NDArray}s in {@link NDList} along first axis.
     *
     * @param arrays the input {@link NDList}. Each {@link NDArray} in the {@link NDList} must have
     *     the same shape as the {@link NDArray}
     * @return the result {@link NDArray}. The stacked {@link NDArray} has one more dimension than
     *     the {@link NDArray}s in {@link NDList}
     */
    default NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a {@link NDArray} along a new axis.
     *
     * @param array the input {@link NDArray} which must have the same shape as the {@link NDArray}
     * @param axis the axis in the result {@link NDArray} along which the input {@link NDArray} are
     *     stacked
     * @return the {@link NDArray}. The stacked {@link NDArray} has one more dimension than the
     *     input {@link NDArray}.
     */
    default NDArray stack(NDArray array, int axis) {
        return stack(new NDList(array), axis);
    }

    /**
     * Joins a {@code NDArray} along first axis.
     *
     * @param array the input {@code NDArray} which must have the same shape as the {@link NDArray}
     * @return the {@code NDArray}. The stacked {@link NDArray} has one more dimension than the
     *     input {@link NDArray}.
     */
    default NDArray stack(NDArray array) {
        return stack(new NDList(array));
    }

    /**
     * Joins a {@link NDList} along an existing axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@link NDArray}, except in
     *     the dimension corresponding to axis
     * @param axis the axis along which the {@link NDList} will be joined
     * @return the concatenated {@link NDArray}
     */
    NDArray concat(NDList arrays, int axis);

    /**
     * Joins a {@link NDList} along first axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@link NDArray}, except in
     *     the dimension corresponding to axis
     * @return the concatenated {@link NDArray}
     */
    default NDArray concat(NDList arrays) {
        return concat(arrays, 0);
    }

    /**
     * Joins a {@link NDArray} along an existing axis.
     *
     * @param array a {@link NDArray} which have the same shape as the {@link NDArray}, except in
     *     the dimension corresponding to axis
     * @param axis the axis along which the {@link NDArray} will be joined
     * @return the concatenated {@link NDArray}
     */
    default NDArray concat(NDArray array, int axis) {
        return concat(new NDList(array), axis);
    }

    /**
     * Joins a {@link NDArray} along first axis.
     *
     * @param array a {@link NDArray} which have the same shape as the {@link NDArray}, except in
     *     the dimension corresponding to axis
     * @return the concatenated {@link NDArray}
     */
    default NDArray concat(NDArray array) {
        return concat(new NDList(array));
    }

    ////////////////////////////////////////
    // Operators: Logical Op
    ////////////////////////////////////////

    /**
     * Returns the truth value of the {@link NDArray} AND the other {@link NDArray} element-wise.
     *
     * @param other the other {@link NDArray} to operate on
     * @return the boolean {@link NDArray} of the logical AND operation applied to the elements of
     *     this {@link NDArray}
     */
    NDArray logicalAnd(NDArray other);

    /**
     * Computes the truth value of the {@link NDArray} OR the other {@link NDArray} element-wise.
     *
     * @param other the other {@link NDArray} to operate on
     * @return the boolean {@link NDArray} of the logical OR operation applied to the elements of
     *     this {@link NDArray}
     */
    NDArray logicalOr(NDArray other);

    /**
     * Computes the truth value of the {@link NDArray} XOR the other {@link NDArray} element-wise..
     *
     * @param other the other {@link NDArray} to operate on
     * @return the boolean {@link NDArray} of the logical XOR operation applied to the elements of
     *     this {@link NDArray}
     */
    NDArray logicalXor(NDArray other);

    /**
     * Computes the truth value of NOT the {@link NDArray} element-wise.
     *
     * @return the boolean {@link NDArray}
     */
    NDArray logicalNot();

    ////////////////////////////////////////
    // Operators: Other
    ////////////////////////////////////////

    /**
     * Performs an indirect sort of the {@link NDArray} ascending on the last axis.
     *
     * @return a {@link NDArray} of indices corresponding to elements in the {@link NDArray} on the
     *     axis, the output DataType is always {@link DataType#INT32}
     * @see NDArray#argsort(int, boolean)
     */
    default NDArray argsort() {
        return argsort(-1, true);
    }

    /**
     * Performs an indirect sort of the {@code NDArray} ascending on the given axis.
     *
     * @param axis the axis to sort along
     * @return a {@code NDArray} of indices corresponding to elements in the {@code NDArray} on the
     *     axis, the output DataType is always {@link DataType#INT32}
     * @see NDArray#argsort(int, boolean)
     */
    default NDArray argsort(int axis) {
        return argsort(axis, true);
    }

    /**
     * Performs an indirect sort of the {@link NDArray} on the given axis.
     *
     * @param axis the axis to sort along
     * @param ascending whether to sort ascending
     * @return a {@link NDArray} of indices corresponding to elements in the {@link NDArray} on the
     *     axis, the output DataType is always {@link DataType#INT32}
     */
    NDArray argsort(int axis, boolean ascending);

    /**
     * Sorts the {@link NDArray} along the given axis.
     *
     * @param axis the axis along which to sort
     * @return the sorted {@link NDArray}
     */
    NDArray sort(int axis);

    /**
     * Sorts the flattened {@link NDArray}.
     *
     * @return the sorted {@link NDArray}
     */
    NDArray sort();

    /**
     * Applies the softmax function along the given axis.
     *
     * @param axis the axis along which to apply
     * @return the result {@link NDArray}
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int[], double)
     */
    default NDArray softmax(int axis) {
        return softmax(new int[] {axis}, 1);
    }

    /**
     * Applies the softmax function along the given axis.
     *
     * @param axis the axis along which to apply
     * @param temperature the exponent multiplier Beta in the softmax
     * @return the result {@link NDArray}
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int[], double)
     */
    default NDArray softmax(int axis, double temperature) {
        return softmax(new int[] {axis}, temperature);
    }

    /**
     * Applies the softmax function along the given axes.
     *
     * @param axes the axes along which to apply. An empty array indicates computing the softmax for
     *     the whole array.
     * @return the result {@link NDArray}
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     */
    default NDArray softmax(int[] axes) {
        return softmax(axes, 1);
    }

    /**
     * Applies the softmax function along the given axes.
     *
     * @param axes the axes to compute the softmax of. An empty array indicates computing the
     *     softmax for the whole array.
     * @param temperature the exponent multiplier Beta in the softmax
     * @return the result {@link NDArray}
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int[], double)
     */
    NDArray softmax(int[] axes, double temperature);

    /**
     * Return the cumulative sum of the elements along a given axis.
     *
     * @param axis the axis along which the cumulative sum is computed
     * @return the cumulative sum along the specified axis
     */
    NDArray cumsum(int axis);

    /**
     * Returns the cumulative sum of the elements in the flattened {@link NDArray}.
     *
     * @return the cumulative sum of the elements in the flattened {@link NDArray}
     */
    NDArray cumsum();

    /**
     * Return the cumulative sum of the elements along a given axis in place.
     *
     * @param axis the axis along which the cumulative sum is computed
     * @return the cumulative sum along the specified axis
     */
    NDArray cumsumi(int axis);

    /**
     * Returns the cumulative sum of the elements in the flattened {@link NDArray} in place.
     *
     * @return the cumulative sum of the elements in the flattened {@link NDArray}
     */
    NDArray cumsumi();

    /**
     * Returns the boolean {@link NDArray} with value {@code true} where this {@link NDArray}'s
     * entries are infinite, or {@code false} where they are not infinite.
     *
     * @return the boolean {@link NDArray} with value {@code true} if the {@link NDArray}'s entries
     *     are infinite
     */
    NDArray isInfinite();

    /**
     * Returns the boolean {@link NDArray} with value {@code true} where this {@link NDArray}'s
     * entries are NaN, or {@code false} where they are not NaN.
     *
     * @return the boolean {@link NDArray} with value {@code true} if the {@link NDArray}'s {@link
     *     NDArray} are NaN
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
     * @return a new boolean {@link NDArray} where values are {@code true} if it matches the
     *     predicate
     */
    NDArray createMask(Predicate<Number> predicate);

    /**
     * Constructs a {@link NDArray} by repeating the {@link NDArray} the number of times given
     * repeats.
     *
     * @param repeats the number of times to repeat for each dimension
     * @return a NDArray that has been tiled
     */
    NDArray tile(long repeats);

    /**
     * Constructs a {@link NDArray} by repeating the {@link NDArray} the number of times given by
     * repeats along given axis.
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return a {@link NDArray} that has been tiled
     * @throws IllegalArgumentException Thrown for invalid axis
     */
    NDArray tile(int axis, long repeats);

    /**
     * Constructs a {@link NDArray} by repeating the {@link NDArray} the number of times given by
     * repeats.
     *
     * @param repeats the number of times to repeat along each axis
     * @return a {@link NDArray} that has been tiled
     */
    NDArray tile(long[] repeats);

    /**
     * Constructs a {@link NDArray} by repeating the {@link NDArray} the number of times to match
     * the desired shape.
     *
     * <p>If the desired shape has fewer dimensions than the {@link NDArray}, it will tile against
     * the last axis.
     *
     * @param desiredShape the shape that should be converted to
     * @return a {@link NDArray} that has been tiled
     */
    NDArray tile(Shape desiredShape);

    /**
     * Repeats element of the {@link NDArray} the number of times given repeats.
     *
     * @param repeats the number of times to repeat for each axis
     * @return an {@link NDArray} that has been repeated
     */
    NDArray repeat(long repeats);

    /**
     * Repeats element of the {@link NDArray} the number of times given repeats along given axis.
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return an {@link NDArray} that has been repeated
     * @throws IllegalArgumentException thrown for invalid axis
     */
    NDArray repeat(int axis, long repeats);

    /**
     * Repeats element of the {@link NDArray} the number of times given repeats along each axis.
     *
     * @param repeats the number of times to repeat along each axis
     * @return an NDArray that has been repeated
     */
    NDArray repeat(long[] repeats);

    /**
     * Repeats element of the {@link NDArray} to match the desired shape.
     *
     * <p>If the desired shape has fewer dimensions that the array, it will tile against the last
     * axis.
     *
     * @param desiredShape the shape that should be converted to
     * @return an NDArray that has been repeated
     */
    NDArray repeat(Shape desiredShape);

    /**
     * Dot product of the {@link NDArray} and the other {@link NDArray}.
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
     * @param other the other {@link NDArray} to perform dot product with
     * @return the result {@link NDArray}
     */
    NDArray dot(NDArray other);

    /**
     * Clips (limit) the values in the {@link NDArray}.
     *
     * <p>Given an interval, values outside the interval are clipped to the interval edges. For
     * example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values
     * larger than 1 become 1.
     *
     * @param min the minimum value double type
     * @param max the maximum value double type
     * @return an {@link NDArray} with the elements of the {@link NDArray}, but where values &lt;
     *     min are replaced with min, and those &gt; max with max
     */
    NDArray clip(double min, double max);

    /**
     * Clips (limit) the values in the {@link NDArray}.
     *
     * <p>Given an interval, values outside the interval are clipped to the interval edges. For
     * example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values
     * larger than 1 become 1.
     *
     * @param min the minimum value int type
     * @param max the maximum value int type
     * @return an {@link NDArray} with the elements of the {@link NDArray}, but where values &lt;
     *     min are replaced with min, and those &gt; max with max
     */
    default NDArray clip(int min, int max) {
        return clip((double) min, (double) max);
    }

    /**
     * Interchanges two axes of the {@link NDArray}.
     *
     * @param axis1 the first axis
     * @param axis2 the second axis
     * @return the swapped axes {@link NDArray}
     */
    default NDArray swapaxes(int axis1, int axis2) {
        int[] dims = IntStream.range(0, getShape().dimension()).toArray();
        int tmp = dims[axis1];
        dims[axis1] = dims[axis2];
        dims[axis2] = tmp;
        return transpose(dims);
    }

    /**
     * Returns the {@link NDArray} with axes transposed.
     *
     * @return the newly permuted array
     */
    NDArray transpose();

    /**
     * Returns the {@link NDArray} with given axes transposed.
     *
     * @param axes the axes to swap to
     * @return the transposed {@link NDArray}
     * @throws IllegalArgumentException thrown when passing a axis that is greater than the actual
     *     number of dimensions
     */
    NDArray transpose(int... axes);

    /**
     * Broadcasts the {@link NDArray} to be the given shape.
     *
     * @param shape the new shape of the {@link NDArray}
     * @return the broadcasted {@link NDArray}
     */
    NDArray broadcast(Shape shape);

    /**
     * Broadcasts the {@link NDArray} to be the given shape.
     *
     * @param shape the new shape of the {@link NDArray}
     * @return the broadcasted {@link NDArray}
     */
    default NDArray broadcast(long... shape) {
        return broadcast(new Shape(shape));
    }

    /**
     * Returns the indices of the maximum values into the flattened {@link NDArray}.
     *
     * @return a {@link NDArray} containing indices
     */
    NDArray argmax();

    /**
     * Returns the indices of the maximum values along given axis.
     *
     * @param axis the axis along which to find maximum values
     * @return a {@link NDArray} containing indices
     */
    NDArray argmax(int axis);

    /**
     * Returns the indices of the minimum values into the flattened {@link NDArray}.
     *
     * @return a {@link NDArray} containing indices
     */
    NDArray argmin();

    /**
     * Returns the indices of the minimum values along given axis.
     *
     * @param axis the axis along which to find minimum values
     * @return a {@link NDArray} containing indices
     */
    NDArray argmin(int axis);

    /**
     * Returns percentile for the {@link NDArray}.
     *
     * @param percentile the target percentile in range of 0..100
     * @return the result {@code NDArray}
     */
    NDArray percentile(Number percentile);

    /**
     * Returns median along given dimension(s).
     *
     * @param percentile the target percentile in range of 0..100
     * @param axes the dimension to calculate percentile for
     * @return the result {@code NDArray} NDArray
     */
    NDArray percentile(Number percentile, int[] axes);

    /**
     * Returns median value for the {@code NDArray}.
     *
     * @return the median {@link NDArray}
     */
    NDArray median();

    /**
     * Returns median value along given axes.
     *
     * @param axes the axes along which to perform the median operation
     * @return the median {@link NDArray} along the specified axes
     */
    NDArray median(int[] axes);

    // ------------ Sparse methods ------------

    /**
     * Returns a dense representation of the sparse {@link NDArray}.
     *
     * @return the result {@link NDArray}
     */
    NDArray toDense();

    /**
     * Returns a sparse representation of {@link NDArray}.
     *
     * @param fmt the {@link SparseFormat} of the {@link NDArray}
     * @return the result {@link NDArray}
     */
    NDArray toSparse(SparseFormat fmt);

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
     * Returns {@code true} if the {@link NDArray} is special case: no-value {@code NDArray}.
     *
     * @return {@code true} if this NDArray is empty
     */
    boolean isEmpty();

    /**
     * Returns {@code true} if all elements within the {@link NDArray} are non-zero or {@code true}.
     *
     * @return {@code true} if all elements within the {@link NDArray} are non-zero or {@code true}
     */
    default boolean all() {
        // result of sum operator is int64 now
        return asType(DataType.BOOLEAN, false).sum().getLong() == size();
    }

    /**
     * Returns {@code true} if any of the elements within the {@link NDArray} are non-zero or {@code
     * true}.
     *
     * @return {@code true} if any of the elements within the {@link NDArray} are non-zero or {@code
     *     true}
     */
    default boolean any() {
        return asType(DataType.BOOLEAN, false).sum().getLong() > 0;
    }

    /**
     * Returns {@code true} if none of the elements within the {@link NDArray} are non-zero or
     * {@code true}.
     *
     * @return {@code true} if none of the elements within the {@link NDArray} are non-zero or
     *     {@code true}
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
     * @return an internal representative of Native {@code NDArray}
     */
    NDArrayEx getNDArrayInternal();

    /** {@inheritDoc} */
    @Override
    void close();
}
