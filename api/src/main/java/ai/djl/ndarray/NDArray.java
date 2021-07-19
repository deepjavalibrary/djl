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
import ai.djl.ndarray.internal.NDFormat;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.Float16Utils;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * An interface representing an n-dimensional array.
 *
 * <p>NDArray is the core data structure for all mathematical computations. An NDArray represents a
 * multidimensional, fixed-size homogeneous array. It has very similar behaviour to the Numpy python
 * package with the addition of efficient computing. To understand how to manage NDArray lifecycle,
 * please refer to <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/docs/development/memory_management.md">NDArray
 * Memory Management Guide</a>
 */
public interface NDArray extends NDResource {

    /**
     * Decodes {@code NDArray} from bytes.
     *
     * @param manager {@link NDManager} used to create this {@code NDArray}
     * @param byteArray data used to decode
     * @return decoded {@code NDArray}
     */
    static NDArray decode(NDManager manager, byte[] byteArray) {
        return manager.decode(byteArray);
    }

    /**
     * Returns the name of this {@code NDArray}.
     *
     * @return the name of this {@code NDArray}
     */
    String getName();

    /**
     * Sets name of this {@code NDArray}.
     *
     * @param name the name of this {@code NDArray}
     */
    void setName(String name);

    /**
     * Returns unique identifier of this {@code NDArray}.
     *
     * @return unique identifier of this {@code NDArray}
     */
    String getUid();

    /**
     * Returns the {@link DataType} of this {@code NDArray}.
     *
     * <p>{@link DataType} is a definition of the precision level of the {@code NDArray}. All values
     * inside the same {@code NDArray} would have the same {@link DataType}.
     *
     * @return the {@link DataType} of this {@code NDArray}
     */
    DataType getDataType();

    /**
     * Returns the {@link Device} of this {@code NDArray}.
     *
     * <p>{@link Device} class contains the information where this {@code NDArray} stored in memory,
     * like CPU/GPU.
     *
     * @return the {@link Device} of this {@code NDArray}
     */
    Device getDevice();

    /**
     * Returns the {@link Shape} of this {@code NDArray}.
     *
     * <p>{@link Shape} defines how this {@code NDArray} is represented multi-dimensionally.
     *
     * @return the {@link Shape} of this {@code NDArray}
     */
    Shape getShape();

    /**
     * Returns the {@link SparseFormat} of this {@code NDArray}.
     *
     * @return the {@link SparseFormat} of this {@code NDArray}
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
     * Returns {@code true} if this {@code NDArray} is a scalar {@code NDArray} with empty {@link
     * Shape}.
     *
     * @return {@code true} if this {@code NDArray} is a scalar {@code NDArray} with empty {@link
     *     Shape}
     */
    default boolean isScalar() {
        return getShape().isScalar();
    }

    /**
     * Encodes {@code NDArray} to byte array.
     *
     * @return byte array
     */
    default byte[] encode() {
        return NDSerializer.encode(this);
    }

    /**
     * Moves this {@code NDArray} to a different {@link Device}.
     *
     * @param device the {@link Device} to be set
     * @param copy set {@code true} if you want to return a copy of the Existing {@code NDArray}
     * @return the result {@code NDArray} with the new {@link Device}
     */
    NDArray toDevice(Device device, boolean copy);

    /**
     * Converts this {@code NDArray} to a different {@link DataType}.
     *
     * @param dataType the {@link DataType} to be set
     * @param copy set {@code true} if you want to return a copy of the Existing {@code NDArray}
     * @return the result {@code NDArray} with the new {@link DataType}
     */
    NDArray toType(DataType dataType, boolean copy);

    /**
     * Attaches a gradient {@code NDArray} to this {@code NDArray} and marks it so {@link
     * ai.djl.training.GradientCollector#backward(NDArray)} can compute the gradient with respect to
     * it.
     *
     * @param requiresGrad if {@code NDArray} requires gradient or not
     */
    void setRequiresGradient(boolean requiresGrad);

    /**
     * Returns the gradient {@code NDArray} attached to this {@code NDArray}.
     *
     * @return the gradient {@code NDArray}
     * @throws NullPointerException when gradient is not initialized
     */
    NDArray getGradient();

    /**
     * Returns true if the gradient calculation is required for this {@code NDArray}.
     *
     * @return true if the gradient calculation is required for this {@code NDArray} else false
     */
    boolean hasGradient();

    /**
     * Returns an NDArray equal to this that stop gradient propagation through it.
     *
     * @return an NDArray equal to this that stops gradient propagation through it
     */
    NDArray stopGradient();

    /**
     * Returns an NDArray equal to this that magnifies the gradient propagated to this by a
     * constant.
     *
     * @param scale how to much to magnify the gradient propagated to this
     * @return an NDArray equal to this that magnifies the gradient propagated to this by a constant
     */
    default NDArray scaleGradient(double scale) {
        return this.mul(scale).add(this.stopGradient().mul(1 - scale));
    }

    /**
     * Returns the size of this {@code NDArray} along a given axis.
     *
     * @param axis the axis to return the size for
     * @return the size of this {@code NDArray} along a given axis
     */
    default long size(int axis) {
        return getShape().size(axis);
    }

    /**
     * Returns the total number of elements in this {@code NDArray}.
     *
     * @return the number of elements in this {@code NDArray}
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
        if (getDataType() == DataType.FLOAT16) {
            return Float16Utils.fromByteBuffer(toByteBuffer());
        } else if (getDataType() != DataType.FLOAT32) {
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
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
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
    default boolean[] toBooleanArray() {
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
     * Converts this {@code NDArray} to a String array.
     *
     * <p>This method is only applicable to the String typed NDArray and not for printing purpose
     *
     * @return Array of Strings
     */
    String[] toStringArray();

    /**
     * Converts this {@code NDArray} to a Number array based on its {@link DataType}.
     *
     * @return a Number array
     */
    default Number[] toArray() {
        switch (getDataType()) {
            case FLOAT16:
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
     * Sets this {@code NDArray} value from {@link Buffer}.
     *
     * @param data the input buffered data
     */
    void set(Buffer data);

    /**
     * Sets this {@code NDArray} value from an array of floats.
     *
     * @param data the array of floats to set
     */
    default void set(float[] data) {
        set(FloatBuffer.wrap(data));
    }

    /**
     * Sets this {@code NDArray} value from an array of ints.
     *
     * @param data the array of integers to set
     */
    default void set(int[] data) {
        set(IntBuffer.wrap(data));
    }

    /**
     * Sets this {@code NDArray} value from an array of doubles.
     *
     * @param data the array of doubles to set
     */
    default void set(double[] data) {
        set(DoubleBuffer.wrap(data));
    }

    /**
     * Sets this {@code NDArray} value from an array of longs.
     *
     * @param data the array of longs to set
     */
    default void set(long[] data) {
        set(LongBuffer.wrap(data));
    }

    /**
     * Sets this {@code NDArray} value from an array of bytes.
     *
     * @param data the array of bytes to set
     */
    default void set(byte[] data) {
        set(ByteBuffer.wrap(data));
    }

    /**
     * Sets the specified index in this {@code NDArray} with the given values.
     *
     * @param index the locations to update
     * @param value the value to replace with. Can broadcast if given smaller dimensions than the
     *     index
     */
    default void set(NDIndex index, NDArray value) {
        getNDArrayInternal().getIndexer().set(this, index, value);
    }

    /**
     * Sets the specified index in this {@code NDArray} with the given value.
     *
     * @param index the locations to update
     * @param value the value to replace with
     */
    default void set(NDIndex index, Number value) {
        getNDArrayInternal().getIndexer().set(this, index, value);
    }

    /**
     * Sets the specific index by a function.
     *
     * @param index the locations to update
     * @param function the function to change the value
     */
    default void set(NDIndex index, Function<NDArray, NDArray> function) {
        NDArray array = get(index);
        set(index, function.apply(array));
    }

    /**
     * Sets the {@code NDArray} by boolean mask.
     *
     * @param index the boolean {@code NDArray} that indicates what to get
     * @param value the value to replace with
     */
    default void set(NDArray index, Number value) {
        set(new NDIndex().addBooleanIndex(index), value);
    }

    /**
     * Sets the specified scalar in this {@code NDArray} with the given value.
     *
     * @param index the single index to update
     * @param value the value to replace with
     * @throws IllegalArgumentException thrown if the index does not correspond to a single element
     */
    default void setScalar(NDIndex index, Number value) {
        getNDArrayInternal().getIndexer().setScalar(this, index, value);
    }

    /**
     * Returns a partial {@code NDArray}.
     *
     * @param index the section of this {@code NDArray} to return
     * @return the partial {@code NDArray}
     */
    default NDArray get(NDIndex index) {
        return getNDArrayInternal().getIndexer().get(this, index);
    }

    /**
     * Returns a partial {@code NDArray}.
     *
     * @param indices the indices used to indicate what to get
     * @param args arguments to replace the varaible "{}" in the indices string. Can be an integer,
     *     long, boolean {@link NDArray}, or integer {@link NDArray}.
     * @return the partial {@code NDArray}
     * @see NDIndex#NDIndex(String, Object...)
     */
    default NDArray get(String indices, Object... args) {
        return get(new NDIndex(indices, args));
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
     * Returns a long element from this {@code NDArray}.
     *
     * @param indices the indices of the long element to return
     * @return the element in the specified index as a long
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default long getLong(long... indices) {
        return getScalar(indices).toLongArray()[0];
    }

    /**
     * Returns a double element from this {@code NDArray}.
     *
     * @param indices the indices of the double element to return
     * @return the element in the specified index as a double
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default double getDouble(long... indices) {
        return getScalar(indices).toDoubleArray()[0];
    }

    /**
     * Returns a float element from this {@code NDArray}.
     *
     * @param indices the indices of the long element to return
     * @return the element in the specified index as a float
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default float getFloat(long... indices) {
        return getScalar(indices).toFloatArray()[0];
    }

    /**
     * Returns an int element from this {@code NDArray}.
     *
     * @param indices the indices of the int element to return
     * @return the element in the specified index as an integer
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default int getInt(long... indices) {
        return getScalar(indices).toIntArray()[0];
    }

    /**
     * Returns an byte element from this {@code NDArray}.
     *
     * @param indices the indices of the byte element to return
     * @return the element in the specified index as a byte
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default byte getByte(long... indices) {
        return getScalar(indices).toByteArray()[0];
    }

    /**
     * Returns an integer element from this {@code NDArray} that represent unsigned integer with 8
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
     * Returns a boolean element from this {@code NDArray}.
     *
     * @param indices the indices of the int element to return
     * @return the element in the specified index as a boolean
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    default boolean getBoolean(long... indices) {
        return getScalar(indices).toBooleanArray()[0];
    }

    /**
     * Deep-copies the current {@code NDArray} to the one passed in.
     *
     * @param array this {@code NDArray} prepared to be copied to
     */
    void copyTo(NDArray array);

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
     * Returns portion of this {@code NDArray} given the index boolean {@code NDArray} along first
     * axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(3, 2));
     * jshell&gt; NDArray mask = manager.create(new boolean[] {true, false, true});
     * jshell&gt; array.booleanMask(mask);
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [5., 6.],
     * ]
     * </pre>
     *
     * @param index boolean {@code NDArray} mask
     * @return the result {@code NDArray}
     */
    default NDArray booleanMask(NDArray index) {
        return booleanMask(index, 0);
    }

    /**
     * Returns portion of this {@code NDArray} given the index boolean {@code NDArray} along given
     * axis.
     *
     * @param index boolean {@code NDArray} mask
     * @param axis an integer that represents the axis of {@code NDArray} to mask from
     * @return the result {@code NDArray}
     */
    NDArray booleanMask(NDArray index, int axis);

    /**
     * Sets all elements outside the sequence to a constant value.
     *
     * <p>This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter {@code
     * sequenceLength} is used to handle variable-length sequences. sequence_length should be an
     * input array of positive ints of dimension [batch_size].
     *
     * @param sequenceLength used to handle variable-length sequences
     * @param value the constant value to be set
     * @return the result {@code NDArray}
     */
    NDArray sequenceMask(NDArray sequenceLength, float value);

    /**
     * Sets all elements outside the sequence to 0.
     *
     * <p>This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter {@code
     * sequenceLength} is used to handle variable-length sequences. sequence_length should be an
     * input array of positive ints of dimension [batch_size].
     *
     * @param sequenceLength used to handle variable-length sequences
     * @return the result {@code NDArray}
     */
    NDArray sequenceMask(NDArray sequenceLength);

    /**
     * Returns an {@code NDArray} of zeros with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; array.zerosLike();
     * ND: (2, 3) cpu() float32
     * [[0., 0., 0.],
     *  [0., 0., 0.],
     * ]
     * </pre>
     *
     * @return a {@code NDArray} filled with zeros
     */
    NDArray zerosLike();

    /**
     * Returns an {@code NDArray} of ones with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; array.onesLike();
     * ND: (2, 3) cpu() float32
     * [[1., 1., 1.],
     *  [1., 1., 1.],
     * ]
     * </pre>
     *
     * @return a {@code NDArray} filled with ones
     */
    NDArray onesLike();

    /**
     * Returns an uninitialized {@code NDArray} with the same {@link Shape}, {@link DataType} and
     * {@link SparseFormat} as the input {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; array.like(); // uninitialized NDArray
     * ND: (2, 3) cpu() float32
     * [[ 9.80908925e-45,  0.00000000e+00,  0.00000000e+00],
     *  [ 0.00000000e+00,  7.61595174e-07,  2.80259693e-44],
     * ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    default NDArray like() {
        return getManager().create(getShape());
    }

    ////////////////////////////////////////
    ////////////////////////////////////////
    // Operations
    ////////////////////////////////////////
    ////////////////////////////////////////

    ////////////////////////////////////////
    // Operations: Element Comparison
    ////////////////////////////////////////

    /**
     * Returns {@code true} if all elements in this {@code NDArray} are equal to the {@link Number}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.ones(new Shape(2, 3));
     * jshell&gt; array.contentEquals(1); // return true instead of boolean NDArray
     * true
     * </pre>
     *
     * @param number the number to compare
     * @return the boolean result
     */
    boolean contentEquals(Number number);

    /**
     * Returns {@code true} if all elements in this {@code NDArray} are equal to the other {@link
     * NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(6f).reshape(2, 3);
     * jshell&gt; NDArray array2 = manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f}, new Shape(2, 3));
     * jshell&gt; array1.contentEquals(array2); // return true instead of boolean NDArray
     * true
     * </pre>
     *
     * @param other the other {@code NDArray} to compare
     * @return the boolean result
     */
    boolean contentEquals(NDArray other);

    /**
     * Checks 2 {@code NDArray}s for equal shapes.
     *
     * <p>Shapes are considered equal if:
     *
     * <ul>
     *   <li>Both {@code NDArray}s have equal rank, and
     *   <li>size(0)...size(rank()-1) are equal for both {@code NDArray}s
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.ones(new Shape(1, 2, 3));
     * jshell&gt; NDArray array2 = manager.create(new Shape(1, 2, 3));
     * jshell&gt; array1.shapeEquals(array2); // return true instead of boolean NDArray
     * true
     * </pre>
     *
     * @param other the other {@code NDArray}
     * @return {@code true} if the {@link Shape}s are the same
     */
    default boolean shapeEquals(NDArray other) {
        return getShape().equals(other.getShape());
    }

    /**
     * Returns {@code true} if two {@code NDArray}s are element-wise equal within a tolerance.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-7});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-8});
     * jshell&gt; array1.allClose(array2); // return false instead of boolean NDArray
     * false
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-8});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-9});
     * jshell&gt; array1.allClose(array2); // return true instead of boolean NDArray
     * true
     * </pre>
     *
     * @param other the {@code NDArray} to compare with
     * @return the boolean result
     */
    default boolean allClose(NDArray other) {
        return allClose(other, 1e-5, 1e-08, false);
    }

    /**
     * Returns {@code true} if two {@code NDArray} are element-wise equal within a tolerance.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-7});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-8});
     * jshell&gt; array1.allClose(array2, 1e-05, 1e-08, false); // return false instead of boolean NDArray
     * false
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-8});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-9});
     * jshell&gt; array1.allClose(array2, 1e-05, 1e-08, false); // return true instead of boolean NDArray
     * true
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, Float.NaN});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, Float.NaN});
     * jshell&gt; array1.allClose(array2, 1e-05, 1e-08, true); // return true instead of boolean NDArray
     * true
     * </pre>
     *
     * @param other the {@code NDArray} to compare with
     * @param rtol the relative tolerance parameter
     * @param atol the absolute tolerance parameter
     * @param equalNan whether to compare NaN’s as equal. If {@code true}, NaN’s in the {@link
     *     NDArray} will be considered equal to NaN’s in the other {@code NDArray}
     * @return the boolean result
     */
    default boolean allClose(NDArray other, double rtol, double atol, boolean equalNan) {
        if (!shapeEquals(other)) {
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
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.ones(new Shape(1));
     * jshell&gt; array.eq(1);
     * ND: (1) cpu() boolean
     * [ true]
     * </pre>
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Equals" comparison
     */
    NDArray eq(Number n);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f, 3f});
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.eq(array2);
     * ND: (3) cpu() boolean
     * [ true,  true, false]
     * </pre>
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Equals" comparison
     */
    NDArray eq(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Not equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2, 2);
     * jshell&gt; array.neq(1);
     * ND: (2, 2) cpu() boolean
     * [[ true, false],
     *  [ true,  true],
     * ]
     * </pre>
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Not equals" comparison
     */
    NDArray neq(Number n);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Not equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 3f});
     * jshell&gt; array1.neq(array2);
     * ND: (2) cpu() boolean
     * [false,  true]
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 3f, 1f, 4f}, new Shape(2, 2));
     * jshell&gt; array1.neq(array2); // broadcasting
     * ND: (2, 2) cpu() boolean
     * [[false,  true],
     *  [false,  true],
     * ]
     * </pre>
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Not equals" comparison
     */
    NDArray neq(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; array.gt(2f);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Greater" comparison
     */
    NDArray gt(Number n);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater Than" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.neq(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wis "Greater Than" comparison
     */
    NDArray gt(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater or equals" comparison.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; array.gte(2f);
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Greater or equals" comparison
     */
    NDArray gte(Number n);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.gte(array2);
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for "Greater or equals" comparison
     */
    NDArray gte(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.lt(2f);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Less" comparison
     */
    NDArray lt(Number n);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.lt(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Less" comparison
     */
    NDArray lt(NDArray other);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.lte(2f);
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Less or equals" comparison
     */
    NDArray lte(Number n);

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.lte(array2);
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Less or equals" comparison
     */
    NDArray lte(NDArray other);

    ////////////////////////////////////////
    // Operations: Element Arithmetic
    ////////////////////////////////////////

    /**
     * Adds a number to this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.add(2f);
     * ND: (2) cpu() float32
     * [3., 4.]
     * </pre>
     *
     * @param n the number to add
     * @return the result {@code NDArray}
     */
    NDArray add(Number n);

    /**
     * Adds other {@code NDArray}s to this {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.add(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[ 0.,  2.,  4.],
     *  [ 3.,  5.,  7.],
     *  [ 6.,  8., 10.],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray}s to add
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray add(NDArray other);

    /**
     * Subtracts a number from this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.sub(2f);
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * </pre>
     *
     * @param n the number to subtract from
     * @return the result {@code NDArray}
     */
    NDArray sub(Number n);

    /**
     * Subtracts the other {@code NDArray} from this {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3);
     * jshell&gt; array1.sub(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0., 0., 0.],
     *  [3., 3., 3.],
     *  [6., 6., 6.],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to subtract from
     * @return the result {@code NDArray}
     */
    NDArray sub(NDArray other);

    /**
     * Multiplies this {@code NDArray} by a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.mul(3f);
     * ND: (2) cpu() float32
     * [3., 6.]
     * </pre>
     *
     * @param n the number to multiply by
     * @return the result {@code NDArray}
     */
    NDArray mul(Number n);

    /**
     * Multiplies this {@code NDArray} by other {@code NDArray}s element-wise.
     *
     * <p>The shapes of this {@code NDArray} and other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.mul(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     *  [ 0.,  4., 10.],
     *  [ 0.,  7., 16.],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray}s to multiply by
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    NDArray mul(NDArray other);

    /**
     * Divides this {@code NDArray} by a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.div(4f);
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * </pre>
     *
     * @param n the number to divide by
     * @return the result {@code NDArray}
     */
    NDArray div(Number n);

    /**
     * Divides this {@code NDArray} by the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.ones(new Shape(3)).mul(10);
     * jshell&gt; array1.div(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0. , 0.1, 0.2],
     *  [0.3, 0.4, 0.5],
     *  [0.6, 0.7, 0.8],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to divide by
     * @return the result {@code NDArray}
     */
    NDArray div(NDArray other);

    /**
     * Returns element-wise remainder of division.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(7f);
     * jshell&gt; array.mod(5f);
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * </pre>
     *
     * @param n the divisor number
     * @return the result {@code NDArray}
     */
    NDArray mod(Number n);

    /**
     * Returns element-wise remainder of division.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 7f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.mod(array2);
     * ND: (2) cpu() float32
     * [0., 1.]
     * </pre>
     *
     * @param other the divisor {@code NDArray}
     * @return the result {@code NDArray}
     */
    NDArray mod(NDArray other);

    /**
     * Takes the power of this {@code NDArray} with a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.pow(4f);
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
     * </pre>
     *
     * @param n the number to take the power with
     * @return the result {@code NDArray}
     */
    NDArray pow(Number n);

    /**
     * Takes the power of this {@code NDArray} with the other {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(6f).reshape(3, 2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.pow(array2); // broadcasting
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     *  [  4.,  27.],
     *  [ 16., 125.],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to take the power with
     * @return the result {@code NDArray}
     */
    NDArray pow(NDArray other);

    /**
     * Adds a number to this {@code NDArray} element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.addi(2f);
     * ND: (2) cpu() float32
     * [3., 4.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 4.]
     * </pre>
     *
     * @param n the number to add
     * @return the result {@code NDArray}
     */
    NDArray addi(Number n);

    /**
     * Adds other {@code NDArray}s to this {@code NDArray} element-wise in place.
     *
     * <p>The shapes of this {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {3f, 4f});
     * jshell&gt; array1.addi(array2);
     * ND: (2) cpu() float32
     * [4., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [4., 6.]
     * </pre>
     *
     * @param other the other {@code NDArray}s to add
     * @return the result {@code NDArray}
     */
    NDArray addi(NDArray other);

    /**
     * Subtracts a number from this {@code NDArray} element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.subi(2f);
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * </pre>
     *
     * @param n the number to subtract
     * @return the result {@code NDArray}
     */
    NDArray subi(Number n);

    /**
     * Subtracts the other {@code NDArray} from this {@code NDArray} element-wise in place.
     *
     * <p>The shapes of this {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.subi(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0., 0., 0.],
     *  [3., 3., 3.],
     *  [6., 6., 6.],
     * ]
     * jshell&gt; array1;
     * [[0., 0., 0.],
     *  [3., 3., 3.],
     *  [6., 6., 6.],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to subtract from
     * @return the result {@code NDArray}
     */
    NDArray subi(NDArray other);

    /**
     * Multiplies this {@code NDArray} by a number element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.muli(3f);
     * ND: (2) cpu() float32
     * [3., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 6.]
     * </pre>
     *
     * @param n the number to multiply by
     * @return the result {@code NDArray}
     */
    NDArray muli(Number n);

    /**
     * Multiplies this {@code NDArray} by other {@code NDArray} element-wise in place.
     *
     * <p>The shapes of this {@code NDArray} and other {@code NDArray}s must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.muli(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     *  [ 0.,  4., 10.],
     *  [ 0.,  7., 16.],
     * ]
     * jshell&gt; array1;
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     *  [ 0.,  4., 10.],
     *  [ 0.,  7., 16.],
     * ]
     * </pre>
     *
     * @param other the other NDArrays to multiply with
     * @return the result {@code NDArray}
     */
    NDArray muli(NDArray other);

    /**
     * Divides this {@code NDArray} by a number element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.divi(4f);
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * </pre>
     *
     * @param n the number to divide values by
     * @return the array after applying division operation
     */
    NDArray divi(Number n);

    /**
     * Divides this {@code NDArray} by the other {@code NDArray} element-wise in place.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.ones(new Shape(3)).mul(10);
     * jshell&gt; array1.divi(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0. , 0.1, 0.2],
     *  [0.3, 0.4, 0.5],
     *  [0.6, 0.7, 0.8],
     * ]
     * jshell&gt; array1;
     * [[0. , 0.1, 0.2],
     *  [0.3, 0.4, 0.5],
     *  [0.6, 0.7, 0.8],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to divide by
     * @return the result of the divide
     */
    NDArray divi(NDArray other);

    /**
     * Returns element-wise remainder of division in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(7f);
     * jshell&gt; array.modi(5f);
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * jshell&gt; array;
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * </pre>
     *
     * @param n the divisor number
     * @return the result {@code NDArray}
     */
    NDArray modi(Number n);

    /**
     * Returns in place element-wise remainder of division in place.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 7f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.modi(array2);
     * ND: (2) cpu() float32
     * [0., 1.]
     * jshell&gt; array1;
     * ND: (2) cpu() float32
     * [0., 1.]
     * </pre>
     *
     * @param other the divisor {@code NDArray}
     * @return the result of the divide
     */
    NDArray modi(NDArray other);

    /**
     * Takes the power of this {@code NDArray} with a number element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.powi(4f);
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
     * jshell&gt; array;
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
     * </pre>
     *
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    NDArray powi(Number n);

    /**
     * Takes the power of this {@code NDArray} with the other {@code NDArray} element-wise in place.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(6f).reshape(3, 2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.powi(array2); // broadcasting
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     *  [  4.,  27.],
     *  [ 16., 125.],
     * ]
     * jshell&gt; array1;
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     *  [  4.,  27.],
     *  [ 16., 125.],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to take the power with
     * @return the result {@code NDArray}
     */
    NDArray powi(NDArray other);

    /**
     * Returns the element-wise sign.
     *
     * @return the result {@code NDArray}
     */
    NDArray sign();

    /**
     * Returns the element-wise sign in-place.
     *
     * @return the result {@code NDArray}
     */
    NDArray signi();

    /**
     * Returns the maximum of this {@code NDArray} and a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; array.maximum(3f);
     * ND: (3) cpu() float32
     * [3., 3., 4.]
     * </pre>
     *
     * @param n the number to be compared
     * @return the maximum of this {@code NDArray} and a number element-wise
     */
    NDArray maximum(Number n);

    /**
     * Returns the maximum of this {@code NDArray} and the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 5f, 2f});
     * jshell&gt; array1.maximum(array2);
     * ND: (3) cpu() float32
     * [2., 5., 4.]
     * jshell&gt; NDArray array1 = manager.eye(2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {0.5f, 2f});
     * jshell&gt; array1.maximum(array2); // broadcasting
     * ND: (2, 2) cpu() float32
     * [[1. , 2. ],
     *  [0.5, 2. ],
     * ]
     * </pre>
     *
     * @param other the {@code NDArray} to be compared
     * @return the maximum of this {@code NDArray} and the other {@code NDArray} element-wise
     */
    NDArray maximum(NDArray other);

    /**
     * Returns the minimum of this {@code NDArray} and a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; array.minimum(3f);
     * ND: (3) cpu() float32
     * [2., 3., 3.]
     * </pre>
     *
     * @param n the number to be compared
     * @return the minimum of this {@code NDArray} and a number element-wise
     */
    NDArray minimum(Number n);

    /**
     * Returns the minimum of this {@code NDArray} and the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 5f, 2f});
     * jshell&gt; array1.minimum(array2);
     * ND: (3) cpu() float32
     * [1., 3., 2.]
     * jshell&gt; NDArray array1 = manager.eye(2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {0.5f, 2f});
     * jshell&gt; array1.minimum(array2); // broadcasting
     * ND: (2, 2) cpu() float32
     * [[0.5, 0. ],
     *  [0. , 1. ],
     * ]
     * </pre>
     *
     * @param other the {@code NDArray} to be compared
     * @return the minimum of this {@code NDArray} and the other {@code NDArray} element-wise
     */
    NDArray minimum(NDArray other);

    ////////////////////////////////////////
    // Operations: Basic Numeric
    ////////////////////////////////////////

    /**
     * Returns the numerical negative {@code NDArray} element-wise.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.neg();
     * ND: (5) cpu() float32
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
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.negi();
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [-0., -1., -2., -3., -4.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray negi();

    /**
     * Returns the absolute value of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1f, -2f});
     * jshell&gt; array.abs();
     * ND: (2) cpu() float32
     * [1., 2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray abs();

    /**
     * Returns the square of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, -3f});
     * jshell&gt; array.square();
     * ND: (2) cpu() float32
     * [4., 9.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray square();

    /**
     * Returns the square root of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {4f});
     * jshell&gt; array.sqrt();
     * ND: (1) cpu() float32
     * [2., ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray sqrt();

    /**
     * Returns the cube-root of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 8f, 27f});
     * jshell&gt; array.cbrt();
     * ND: (3) cpu() float32
     * [1., 2., 3.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray cbrt();

    /**
     * Returns the floor of this {@code NDArray} element-wise.
     *
     * <p>The floor of the scalar x is the largest integer i, such that i &lt;= x.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.floor();
     * ND: (7) cpu() float32
     * [-2., -2., -1.,  0.,  1.,  1.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray floor();

    /**
     * Returns the ceiling of this {@code NDArray} element-wise.
     *
     * <p>The ceil of the scalar x is the smallest integer i, such that i &gt;= x.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.ceil();
     * ND: (7) cpu() float32
     * [-1., -1., -0.,  1.,  2.,  2.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray ceil();

    /**
     * Returns the round of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.round();
     * ND: (7) cpu() float32
     * [-2., -2., -0.,  0.,  2.,  2.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray round();

    /**
     * Returns the truncated value of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.trunc();
     * ND: (7) cpu() float32
     * [-1., -1., -0.,  0.,  1.,  1.,  2.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray trunc();

    /**
     * Returns the exponential value of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 2.5f});
     * jshell&gt; array.exp();
     * ND: (2) cpu() float32
     * [ 1.    , 12.1825]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray exp();

    /**
     * Returns the natural logarithmic value of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 2.5f});
     * jshell&gt; array.log();
     * ND: (2) cpu() float32
     * [  -inf, 0.9163]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray log();

    /**
     * Returns the base 10 logarithm of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1000f, 1f, 150f});
     * jshell&gt; array.log10();
     * ND: (3) cpu() float32
     * [3.    , 0.    , 2.1761]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray log10();

    /**
     * Returns the base 2 logarithm of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {8, 1f, 5f});
     * jshell&gt; array.log2();
     * ND: (3) cpu() float32
     * [3.    , 0.    , 2.3219]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray log2();

    /**
     * Returns the trigonometric sine of this {@code NDArray} element-wise.
     *
     * <p>The input should be in radians (2 Pi radians equals 360 degrees).
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 30f, 45f, 60f, 90f});
     * jshell&gt; array = array.mul(Math.PI).div(180f);
     * jshell&gt; array.sin();
     * ND: (5) cpu() float32
     * [0.    , 0.5   , 0.7071, 0.866 , 1.    ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray sin();

    /**
     * Returns the trigonometric cosine of this {@code NDArray} element-wise.
     *
     * <p>The input should be in radians (2 Pi radians equals 360 degrees).
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI/2, Math.PI});
     * jshell&gt; array.cos();
     * ND: (3) cpu() float64
     * [  1.0000000e+00,   6.1232340e-17,  -1.0000000e+00],
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray cos();

    /**
     * Returns the trigonometric tangent of this {@code NDArray} element-wise.
     *
     * <p>The input should be in radians (2 Pi radians equals 360 degrees).
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {-Math.PI, Math.PI/2, Math.PI});
     * jshell&gt; array.tan();
     * ND: (3) cpu() float64
     * [  1.2246468e-16,   1.6331239e+16,  -1.2246468e-16],
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray tan();

    /**
     * Returns the inverse trigonometric sine of this {@code NDArray} element-wise.
     *
     * <p>The input should be in the range [-1, 1]. The output is in the closed interval of [-Pi/2,
     * Pi/2].
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, -1f, 0f});
     * jshell&gt; array.asin();
     * ND: (3) cpu() float64
     * [ 1.5708, -1.5708,  0.    ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray asin();

    /**
     * Returns the inverse trigonometric cosine of this {@code NDArray} element-wise.
     *
     * <p>The input should be in the range [-1, 1]. The output is in the closed interval of [-Pi/2,
     * Pi/2].
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, -1f});
     * jshell&gt; array.acos();
     * ND: (2) cpu() float64
     * [0.    , 3.1416]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray acos();

    /**
     * Returns the inverse trigonometric tangent of this {@code NDArray} element-wise.
     *
     * <p>The input should be in the range [-1, 1]. The output is in the closed interval of [-Pi/2,
     * Pi/2].
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f});
     * jshell&gt; array.atan();
     * ND: (2) cpu() float64
     * [0.    , 0.7854]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray atan();

    /**
     * Returns the hyperbolic sine of this {@code NDArray} element-wise.
     *
     * <p>sinh(x)=0.5*(exp(x) - exp(-x))
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.sinh();
     * ND: (2) cpu() float64
     * [ 0.    , 11.5487]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray sinh();

    /**
     * Returns the hyperbolic cosine of this {@code NDArray} element-wise.
     *
     * <p>cosh(x)=0.5*(exp(x)+exp(-x))
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.cosh();
     * ND: (2) cpu() float64
     * [ 1.    , 11.592 ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray cosh();

    /**
     * Returns the hyperbolic tangent of this {@code NDArray} element-wise.
     *
     * <p>tanh(x)=sinh(x)/cosh(x)
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.tanh();
     * ND: (2) cpu() float64
     * [0.    , 0.9963]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray tanh();

    /**
     * Returns the inverse hyperbolic sine of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {Math.E, 10});
     * jshell&gt; array.asinh();
     * ND: (2) cpu() float64
     * [1.7254, 2.9982]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray asinh();

    /**
     * Returns the inverse hyperbolic cosine of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {Math.E, 10});
     * jshell&gt; array.acosh();
     * ND: (2) cpu() float64
     * [1.6575, 2.9932]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray acosh();

    /**
     * Returns the inverse hyperbolic tangent of this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, -0.5});
     * jshell&gt; array.atanh();
     * ND: (2) cpu() float64
     * [ 0.    , -0.5493]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray atanh();

    /**
     * Converts this {@code NDArray} from radians to degrees element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).mul(Math.PI / 3);
     * jshell&gt; array.toDegrees();
     * ND: (6) cpu() float32
     * [  0.,  60., 120., 180., 240., 300.]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray toDegrees();

    /**
     * Converts this {@code NDArray} from degrees to radians element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).mul(60);
     * jshell&gt; array.toRadians();
     * ND: (6) cpu() float32
     * [0.    , 1.0472, 2.0944, 3.1416, 4.1888, 5.236 ]
     * </pre>
     *
     * @return the result {@code NDArray}
     */
    NDArray toRadians();

    ////////////////////////////////////////
    // Operations: Reduction
    ////////////////////////////////////////

    /**
     * Returns the maximum of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.max(); // Maximum of the flattened array
     * ND: () cpu() float32
     * 3.
     * jshell&gt; array.max().getFloat() // Use getFloat() to get native float
     * 3.0
     * </pre>
     *
     * @return the maximum of this {@code NDArray}
     */
    NDArray max();

    /**
     * Returns the maximum of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{0}); // Maximum along the first axis
     * ND: (2) cpu() float32
     * [2., 3.]
     * jshell&gt; array.max(new int[]{1}); // Maximum along the second axis
     * ND: (2) cpu() float32
     * [1., 3.]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @return the maximum of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the max
     * @see NDArray#max(int[], boolean)
     */
    default NDArray max(int[] axes) {
        return max(axes, false);
    }

    /**
     * Returns the maximum of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{0}, true); // Maximum along the first axis and keep dimension
     * ND: (1, 2) cpu() float32
     * [[2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{1}, true); // Maximum along the second axis and keep dimension
     * ND: (2, 1) cpu() float32
     * [[1.],
     *  [3.],
     * ]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array.
     * @return the maximum of this {@code NDArray}
     */
    NDArray max(int[] axes, boolean keepDims);

    /**
     * Returns the minimum of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.min(); // Minimum of the flattened array
     * ND: () cpu() float32
     * 0.
     * jshell&gt; array.min().getFloat(); // Use getFloat() to get native float
     * 0.0
     * </pre>
     *
     * @return the minimum of this {@code NDArray}
     */
    NDArray min();

    /**
     * Returns the minimum of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.min(new int[]{0}); // Minimum along the first axis
     * ND: (2) cpu() float32
     * [0., 1.]
     * jshell&gt; array.min(new int[]{1}); // Minimum along the second axis
     * ND: (2) cpu() float32
     * [0., 2.]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @return the minimum of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the min
     * @see NDArray#min(int[], boolean)
     */
    default NDArray min(int[] axes) {
        return min(axes, false);
    }

    /**
     * Returns the minimum of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array.min(new int[]{0}, true) // Minimum along the first axis and keep dimension
     * ND: (1, 2) cpu() float32
     * [[0., 1.],
     * ]
     * jshell&gt; array.min(new int[]{1}, true) // Minimum along the second axis and keep dimension
     * ND: (2, 1) cpu() float32
     * [[0.],
     *  [2.],
     * ]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the minimum of this {@code NDArray}
     */
    NDArray min(int[] axes, boolean keepDims);

    /**
     * Returns the sum of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0.5f, 1.5f});
     * jshell&gt; array.sum();
     * ND: () cpu() float32
     * 2.
     * jshell&gt; array.sum().getFloat(); // Use getFloat() to get native float
     * 2.0
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 0f, 5f}, new Shape(2, 2));
     * jshell&gt; array.sum();
     * ND: () cpu() float32
     * 6.
     * </pre>
     *
     * @return the sum of this {@code NDArray}
     */
    NDArray sum();

    /**
     * Returns the sum of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 0f, 5f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [0., 5.],
     * ]
     * jshell&gt; array.sum(new int[] {0});
     * ND: (2) cpu() float32
     * [0., 6.]
     * jshell&gt; array.sum(new int[] {1});
     * ND: (2) cpu() float32
     * [1., 5.]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @return the sum of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the sum
     * @see NDArray#sum(int[], boolean)
     */
    default NDArray sum(int[] axes) {
        return sum(axes, false);
    }

    /**
     * Returns the sum of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 0f, 5f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [0., 5.],
     * ]
     * jshell&gt; array.sum(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[0., 6.],
     * ]
     * jshell&gt; array.sum(new int[] {1}, true);
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [0., 5.],
     * ]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the sum of this {@code NDArray}
     */
    NDArray sum(int[] axes, boolean keepDims);

    /**
     * Returns the product of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f});
     * jshell&gt; array.prod();
     * ND: () cpu() float32
     * 6.
     * jshell&gt; array.prod().getFloat(); // Use getFloat to get native float
     * 6.0
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.prod();
     * ND: () cpu() float32
     * 24.
     * </pre>
     *
     * @return the product of this {@code NDArray}
     */
    NDArray prod();

    /**
     * Returns the product of this {@code NDArray} elements over the given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.prod(new int[] {0});
     * ND: (2) cpu() float32
     * [3., 8.]
     * jshell&gt; array.prod(new int[] {1});
     * ND: (2) cpu() float32
     * [ 2., 12.]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @return the product of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the prod
     * @see NDArray#prod(int[], boolean)
     */
    default NDArray prod(int[] axes) {
        return prod(axes, false);
    }

    /**
     * Returns the product of this {@code NDArray} elements over the given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.prod(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[3., 8.],
     * ]
     * jshell&gt; array.prod(new int[] {1}, true);
     * ND: (2, 1) cpu() float32
     * [[ 2.],
     *  [12.],
     * ]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the product of this {@code NDArray}
     */
    NDArray prod(int[] axes, boolean keepDims);

    /**
     * Returns the average of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f});
     * jshell&gt; array.mean();
     * ND: () cpu() float32
     * 2.5
     * jshell&gt; array.mean().getFloat(); // Use getFloat() to get native float
     * 2.5
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.mean();
     * ND: () cpu() float32
     * 2.5
     * </pre>
     *
     * @return the average of this {@code NDArray}
     */
    NDArray mean();

    /**
     * Returns the average of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.mean(new int[] {0});
     * ND: (2) cpu() float32
     * [2., 3.]
     * jshell&gt; array.mean(new int[] {1});
     * ND: (2) cpu() float32
     * [1.5, 3.5]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @return the average of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the mean
     * @see NDArray#mean(int[], boolean)
     */
    default NDArray mean(int[] axes) {
        return mean(axes, false);
    }

    /**
     * Returns the average of this {@code NDArray} along given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.mean(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[2., 3.],
     * ]
     * jshell&gt; array.mean(new int[] {1}, true);
     * ND: (2, 1) cpu() float32
     * [[1.5],
     *  [3.5],
     * ]
     * </pre>
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the average of this {@code NDArray}
     */
    NDArray mean(int[] axes, boolean keepDims);

    /**
     * Rotates an array by 90 degrees in the plane specified by axes.
     *
     * <p>Rotation direction is from the first towards the second axis.
     *
     * @param times Number of times the array is rotated by 90 degrees.
     * @param axes The array is rotated in the plane defined by the axes. Axes must be different.
     * @return the rotated NDArray
     */
    NDArray rotate90(int times, int[] axes);

    /**
     * Returns the sum along diagonals of this {@code NDArray}.
     *
     * <p>If this {@code NDArray} is 2-D, the sum along its diagonal is returned. If the {@link
     * NDArray} has more than two dimensions, then the axes specified by axis1 and axis2 are used to
     * determine the 2-D sub-arrays whose traces are returned. The {@link Shape} of the resulting
     * {@link NDArray} is the same as that of a with axis1 and axis2 removed.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.eye(3);
     * jshell&gt; array;
     * ND: (3, 3) cpu() float32
     * [[1., 0., 0.],
     *  [0., 1., 0.],
     *  [0., 0., 1.],
     * ]
     * jshell&gt; array.trace();
     * ND: () cpu() float32
     * 3.
     * jshell&gt; NDArray array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     *   [2., 3.],
     *  ],
     *  [[4., 5.],
     *   [6., 7.],
     *  ],
     * ]
     * jshell&gt; array.trace();
     * ND: (2) cpu() float32
     * [6., 8.]
     * </pre>
     *
     * @return the sum along diagonals of this {@code NDArray}
     */
    default NDArray trace() {
        return trace(0, 0, 1);
    }

    /**
     * Returns the sum along diagonals of this {@code NDArray}.
     *
     * <p>If this {@code NDArray} is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If this {@code NDArray} has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The {@link Shape} of the resulting array is the same as
     * this {@code NDArray} with axis1 and axis2 removed.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.eye(3);
     * jshell&gt; array;
     * ND: (3, 3) cpu() float32
     * [[1., 0., 0.],
     *  [0., 1., 0.],
     *  [0., 0., 1.],
     * ]
     * jshell&gt; array.trace(1);
     * ND: () cpu() float32
     * 0.
     * jshell&gt; NDArray array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     *   [2., 3.],
     *  ],
     *  [[4., 5.],
     *   [6., 7.],
     *  ],
     * ]
     * jshell&gt; array.trace(1);
     * ND: (2) cpu() float32
     * [2., 3.]
     * </pre>
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @return the sum along diagonals of this {@code NDArray}
     */
    default NDArray trace(int offset) {
        return trace(offset, 0, 1);
    }

    /**
     * Returns the sum along diagonals of this {@code NDArray}.
     *
     * <p>If this {@code NDArray} is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If this {@code NDArray} has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The {@link Shape} of the resulting array is the same as
     * this {@code NDArray} with axis1 and axis2 removed.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     *   [2., 3.],
     *  ],
     *  [[4., 5.],
     *   [6., 7.],
     *  ],
     * ]
     * jshell&gt; array.trace(1,1,2);
     * ND: (2) cpu() float32
     * [1., 5.]
     * </pre>
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @param axis1 axes to be used as the first axis of the 2-D sub-arrays from which the diagonals
     *     should be taken
     * @param axis2 axes to be used as the second axis of the 2-D sub-arrays from which the
     *     diagonals should be taken
     * @return the sum along diagonals of this {@code NDArray}
     */
    NDArray trace(int offset, int axis1, int axis2);

    ////////////////////////////////////////
    // Operations: Shapes and Arrays Manipulation
    ////////////////////////////////////////

    /**
     * Splits this {@code NDArray} into multiple sub{@code NDArray}s given sections along first
     * axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(9f);
     * jshell&gt; array.split(3).forEach(System.out::println);
     * ND: (3) cpu() float32
     * [0., 1., 2.]
     *
     * ND: (3) cpu() float32
     * [3., 4., 5.]
     *
     * ND: (3) cpu() float32
     * [6., 7., 8.]
     * </pre>
     *
     * @param sections this {@code NDArray} will be divided into N (sections) equal {@code NDArray}
     * @return an {@link NDList} with size(axis) {@code NDArray}s with {@link Shape} {@code
     *     this.shape.remove(axis) }
     * @see NDArray#split(long, int)
     */
    default NDList split(long sections) {
        return split(sections, 0);
    }

    /**
     * Splits this {@code NDArray} into multiple sub-{@code NDArray}s given indices along first
     * axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(8f);
     * jshell&gt; array.split(new int[] {3, 5, 6}).forEach(System.out::println);
     * ND: (3) cpu() float32
     * [0., 1., 2.]
     *
     * ND: (2) cpu() float32
     * [3., 4.]
     *
     * ND: (1) cpu() float32
     * [5.]
     *
     * ND: (2) cpu() float32
     * [6., 7.]
     * </pre>
     *
     * @param indices the entries indicate where along axis this {@code NDArray} is split. If an
     *     index exceeds the dimension of this {@code NDArray} along axis, an empty sub-{@link
     *     NDArray} is returned correspondingly.
     * @return an NDList with size(axis) {@code NDArray}s with {@link Shape} {@code
     *     this.shape.remove(axis) }
     * @see NDArray#split(long[], int)
     */
    default NDList split(long[] indices) {
        return split(indices, 0);
    }

    /**
     * Splits this {@code NDArray} into multiple sub{@code NDArray}s given sections along the given
     * axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(18f).reshape(2, 9);
     * jshell&gt; array;
     * ND: (2, 9) cpu() float32
     * [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
     *  [ 9., 10., 11., 12., 13., 14., 15., 16., 17.],
     * ]
     * jshell&gt; array.split(3, 1).forEach(System.out::println);
     * ND: (2, 3) cpu() float32
     * [[ 0.,  1.,  2.],
     *  [ 9., 10., 11.],
     * ]
     *
     * ND: (2, 3) cpu() float32
     * [[ 3.,  4.,  5.],
     *  [12., 13., 14.],
     * ]
     *
     * ND: (2, 3) cpu() float32
     * [[ 6.,  7.,  8.],
     *  [15., 16., 17.],
     * ]
     * </pre>
     *
     * @param sections this {@code NDArray} will be divided into N (sections) equal arrays along
     *     axis
     * @param axis the axis to split along
     * @return an {@link NDList} with numOutputs {@code NDArray}s with {@link Shape} {@code
     *     (this.shape.axis /= axis) }
     * @throws IllegalArgumentException thrown if the numOutputs does not equally divide the given
     *     axis
     */
    default NDList split(long sections, int axis) {
        long axisSize = getShape().getShape()[axis];
        if (axisSize % sections != 0) {
            throw new IllegalArgumentException("array split does not result in an equal division");
        }
        long sectionSize = axisSize / sections;
        long[] indices = LongStream.range(0, sections).map(i -> i * sectionSize).toArray();
        return split(indices, axis);
    }

    /**
     * Splits this {@code NDArray} into multiple sub-{@code NDArray}s given indices along given
     * axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(18f).reshape(2, 9);
     * jshell&gt; array;
     * ND: (2, 9) cpu() float32
     * [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
     *  [ 9., 10., 11., 12., 13., 14., 15., 16., 17.],
     * ]
     * jshell&gt; array.split(new int[] {2,4,5}, 1).forEach(System.out::println);
     * ND: (2, 2) cpu() float32
     * [[ 0.,  1.],
     *  [ 9., 10.],
     * ]
     *
     * ND: (2, 2) cpu() float32
     * [[ 2.,  3.],
     *  [11., 12.],
     * ]
     *
     * ND: (2, 1) cpu() float32
     * [[ 4.],
     *  [13.],
     * ]
     *
     * ND: (2, 4) cpu() float32
     * [[ 5.,  6.,  7.,  8.],
     *  [14., 15., 16., 17.],
     * ]
     * </pre>
     *
     * @param indices the entries indicate where along axis this {@code NDArray} is split. If an
     *     index exceeds the dimension of this {@code NDArray} along axis, an empty sub-array is
     *     returned correspondingly
     * @param axis the axis to split along
     * @return an {@link NDList} with numOutputs {@code NDArray}s with {@link Shape} {@code
     *     (this.shape.axis /= axis) }
     */
    NDList split(long[] indices, int axis);

    /**
     * Flattens this {@code NDArray} into a 1-D {@code NDArray} in row-major order.
     *
     * <p>To flatten in column-major order, first transpose this {@code NDArray}
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[]{1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.flatten();
     * ND: (4) cpu() float32
     * [1., 2., 3., 4.]
     * </pre>
     *
     * @return a 1-D {@code NDArray} of equal size
     */
    NDArray flatten();

    /**
     * Reshapes this {@code NDArray} to the given {@link Shape}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f);
     * jshell&gt; array;
     * ND: (6) cpu() float32
     * [0., 1., 2., 3., 4., 5.]
     * jshell&gt; array.reshape(2, 3);
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * </pre>
     *
     * @param newShape the long array to reshape into. Must have equal size to the current shape
     * @return a reshaped {@code NDArray}
     * @throws IllegalArgumentException thrown if the given {@link Shape} does not match the size of
     *     the current shape
     */
    default NDArray reshape(long... newShape) {
        return reshape(new Shape(newShape));
    }

    /**
     * Reshapes this {@code NDArray} to the given {@link Shape}.
     *
     * <p>You can reshape it to match another NDArray by calling {@code a.reshape(b.getShape()) }
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f);
     * jshell&gt; array;
     * ND: (6) cpu() float32
     * [0., 1., 2., 3., 4., 5.]
     * jshell&gt; array.reshape(new Shape(2, 3));
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * </pre>
     *
     * @param shape the {@link Shape} to reshape into. Must have equal size to the current shape
     * @return a reshaped {@code NDArray}
     * @throws IllegalArgumentException thrown if the given {@link Shape} does not match the size of
     *     the current shape
     */
    NDArray reshape(Shape shape);

    /**
     * Expands the {@link Shape} of a {@code NDArray}.
     *
     * <p>Inserts a new axis that will appear at the axis position in the expanded {@code NDArray}
     * shape.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array.expandDims(0);
     * ND: (1, 2) cpu() float32
     * [[1., 2.],
     * ]
     * jshell&gt; array.expandDims(1);
     * ND: (2, 1) cpu() float32
     * [[1.],
     *  [2.],
     * ]
     * </pre>
     *
     * @param axis the position in the expanded axes where the new axis is placed
     * @return the result {@code NDArray}. The number of dimensions is one greater than that of the
     *     {@code NDArray}
     */
    NDArray expandDims(int axis);

    /**
     * Removes all singleton dimensions from this {@code NDArray} {@link Shape}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f}, new Shape(1, 3, 1));
     * jshell&gt; array;
     * ND: (1, 3, 1) cpu() float32
     * [[[0.],
     *   [1.],
     *   [2.],
     *  ],
     * ]
     * jshell&gt; array.squeeze();
     * ND: (3) cpu() float32
     * [0., 1., 2.]
     * </pre>
     *
     * @return a result {@code NDArray} of same size and data without singleton dimensions
     */
    default NDArray squeeze() {
        long[] shape = getShape().getShape();
        return squeeze(IntStream.range(0, shape.length).filter(i -> shape[i] == 1).toArray());
    }

    /**
     * Removes a singleton dimension at the given axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f}, new Shape(1, 3, 1));
     * jshell&gt; array;
     * ND: (1, 3, 1) cpu() float32
     * [[[0.],
     *   [1.],
     *   [2.],
     *  ],
     * ]
     * jshell&gt; array.squeeze(0);
     * ND: (3, 1) cpu() float32
     * [[0.],
     *  [1.],
     *  [2.],
     * ]
     * jshell&gt; array.squeeze(2);
     * ND: (1, 3) cpu() float32
     * [[0., 1., 2.],
     * ]
     * </pre>
     *
     * @param axis the axis at which to remove the singleton dimension
     * @return a result {@code NDArray} of same size and data without the axis at part of the shape
     * @throws IllegalArgumentException thrown if the given axis is not a singleton dimension
     */
    default NDArray squeeze(int axis) {
        return squeeze(new int[] {axis});
    }

    /**
     * Removes singleton dimensions at the given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f}, new Shape(1, 3, 1));
     * jshell&gt; array;
     * ND: (1, 3, 1) cpu() float32
     * [[[0.],
     *   [1.],
     *   [2.],
     *  ],
     * ]
     * jshell&gt; array.squeeze(new int[] {0, 2});
     * ND: (3) cpu() float32
     * [0., 1., 2.]
     * </pre>
     *
     * @param axes the axes at which to remove the singleton dimensions
     * @return a result {@code NDArray} of same size and data without the axes at part of the shape
     * @throws IllegalArgumentException thrown if any of the given axes are not a singleton
     *     dimension
     */
    NDArray squeeze(int[] axes);

    /**
     * Joins a {@code NDArray} along the first axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.stack(array2)
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * </pre>
     *
     * @param array the input {@code NDArray} which must have the same {@link Shape}as this {@code
     *     NDArray}
     * @return the result {@code NDArray}. The stacked {@code NDArray} has one more dimension than
     *     the input {@code NDArray}.
     */
    default NDArray stack(NDArray array) {
        return stack(array, 0);
    }

    /**
     * Joins a {@code NDArray} along a new axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.stack(array2, 0);
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     *  [2., 3.],
     * ]
     * jshell&gt; array1.stack(array2, 1);
     * ND: (2, 2) cpu() float32
     * [[0., 2.],
     *  [1., 3.],
     * ]
     * </pre>
     *
     * @param array the input {@code NDArray} which must have the same {@link Shape}as this {@code
     *     NDArray}
     * @param axis the axis in the result {@code NDArray} along which the input {@code NDArray} are
     *     stacked
     * @return the result {@code NDArray}. The stacked {@code NDArray} has one more dimension than
     *     the input {@code NDArray}.
     */
    default NDArray stack(NDArray array, int axis) {
        return getNDArrayInternal().stack(new NDList(array), axis);
    }

    /**
     * Joins a {@code NDArray} along the first axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.concat(array2)
     * ND: (4) cpu() float32
     * [0., 1., 2., 3.]
     * </pre>
     *
     * @param array a {@code NDArray} which have the same {@link Shape}as this {@code NDArray},
     *     except in the dimension corresponding to axis
     * @return the concatenated {@code NDArray}
     */
    default NDArray concat(NDArray array) {
        return concat(array, 0);
    }

    /**
     * Joins a {@code NDArray} along an existing axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.concat(array2, 0);
     * ND: (4) cpu() float32
     * [0., 1., 2., 3.]
     * </pre>
     *
     * @param array a {@code NDArray} which have the same {@link Shape}as this {@code NDArray},
     *     except in the dimension corresponding to axis
     * @param axis the axis along which this {@code NDArray} will be joined
     * @return the concatenated {@code NDArray}
     */
    default NDArray concat(NDArray array, int axis) {
        return getNDArrayInternal().concat(new NDList(array), axis);
    }

    ////////////////////////////////////////
    // Operations: Logical Op
    ////////////////////////////////////////

    /**
     * Returns the truth value of this {@code NDArray} AND the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new boolean[] {true});
     * jshell&gt; NDArray array2 = manager.create(new boolean[] {false});
     * jshell&gt; array1.logicalAnd(array2);
     * ND: (1) cpu() boolean
     * [false]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; array1.logicalAnd(array2);
     * ND: (2) cpu() boolean
     * [false, false]
     * </pre>
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.gt(1).logicalAnd(array.lt(4));
     * ND: (5) cpu() boolean
     * [false, false,  true,  true, false]
     * </pre>
     *
     * @param other the other {@code NDArray} to operate on
     * @return the boolean {@code NDArray} of the logical AND operation applied to the elements of
     *     this {@code NDArray} and the other {@code NDArray}
     */
    NDArray logicalAnd(NDArray other);

    /**
     * Computes the truth value of this {@code NDArray} OR the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new boolean[] {true});
     * jshell&gt; NDArray array2 = manager.create(new boolean[] {false});
     * jshell&gt; array1.logicalOr(array2);
     * ND: (1) cpu() boolean
     * [ true]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; array1.logicalOr(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.lt(1).logicalOr(array.gt(3));
     * ND: (5) cpu() boolean
     * [ true, false, false, false,  true]
     * </pre>
     *
     * @param other the other {@code NDArray} to operate on
     * @return the boolean {@code NDArray} of the logical OR operation applied to the elements of
     *     this {@code NDArray} and the other {@code NDArray}
     */
    NDArray logicalOr(NDArray other);

    /**
     * Computes the truth value of this {@code NDArray} XOR the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true});
     * jshell&gt; array1.logicalXor(array2);
     * ND: (1) cpu() boolean
     * [ true]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; array1.logicalXor(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.lt(1).logicalXor(array.gt(3));
     * ND: (5) cpu() boolean
     * [ true, false, false, false,  true]
     * </pre>
     *
     * @param other the other {@code NDArray} to operate on
     * @return the boolean {@code NDArray} of the logical XOR operation applied to the elements of
     *     this {@code NDArray} and the other {@code NDArray}
     */
    NDArray logicalXor(NDArray other);

    /**
     * Computes the truth value of NOT this {@code NDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true});
     * jshell&gt; array.logicalNot();
     * ND: (1) cpu() boolean
     * [ false]
     * </pre>
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.lt(1).logicalNot();
     * ND: (5) cpu() boolean
     * [false, true, true,  true,  true]
     * </pre>
     *
     * @return the boolean {@code NDArray}
     */
    NDArray logicalNot();

    ////////////////////////////////////////
    // Operations: Other
    ////////////////////////////////////////

    /**
     * Returns the indices that would sort this {@code NDArray}.
     *
     * <p>Perform an indirect sort along the given axis. It returns a {@code NDArray} of indices of
     * the same {@link Shape} as this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {3f, 1f, 2f});
     * jshell&gt; array.argSort();
     * ND: (3) cpu() int64
     * [ 1,  2,  0]
     *
     * jshell&gt; array = manager.create(new float[] {0f, 3f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array.argSort();
     * ND: (2, 2) cpu() int64
     * [[ 0,  1],
     *  [ 0,  1],
     * ]
     * </pre>
     *
     * @return a {@code NDArray} of indices corresponding to elements in this {@code NDArray} on the
     *     axis, the output DataType is always {@link DataType#INT64}
     * @see NDArray#argSort(int, boolean)
     */
    default NDArray argSort() {
        return argSort(-1, true);
    }

    /**
     * Returns the indices that would sort this {@code NDArray} given the axis.
     *
     * <p>Perform an indirect sort along the given axis. It returns a {@code NDArray} of indices of
     * the same {@link Shape} as this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 3f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array.argSort(0);
     * ND: (2, 2) cpu() int64
     * [[ 0,  1],
     *  [ 1,  0],
     * ]
     * jshell&gt; array.argSort(1);
     * ND: (2, 2) cpu() int64
     * [[ 0,  1],
     *  [ 0,  1],
     * ]
     * </pre>
     *
     * @param axis the axis to sort along
     * @return a {@code NDArray} of indices corresponding to elements in this {@code NDArray} on the
     *     axis, the output DataType is always {@link DataType#INT64}
     * @see NDArray#argSort(int, boolean)
     */
    default NDArray argSort(int axis) {
        return argSort(axis, true);
    }

    /**
     * Returns the indices that would sort this {@code NDArray} given the axis.
     *
     * <p>Perform an indirect sort along the given axis. It returns a {@code NDArray} of indices of
     * the same {@link Shape} as this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 3f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array.argSort(0, false);
     * ND: (2, 2) cpu() int64
     * [[ 1,  0],
     *  [ 0,  1],
     * ]
     * </pre>
     *
     * @param axis the axis to sort along
     * @param ascending whether to sort ascending
     * @return a {@code NDArray} of indices corresponding to elements in this {@code NDArray} on the
     *     axis, the output DataType is always {@link DataType#INT64}
     */
    NDArray argSort(int axis, boolean ascending);

    /**
     * Sorts the flattened {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 4f, 3f, 1f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 4.],
     *  [3., 1.],
     * ]
     * jshell&gt; array.sort(); // sort the flattened array
     * ND: (2, 2) cpu() float32
     * [[1., 4.],
     *  [1., 3.],
     * ]
     * </pre>
     *
     * @return the sorted {@code NDArray}
     */
    NDArray sort();

    /**
     * Sorts the flattened {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 4f, 3f, 1f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 4.],
     *  [3., 1.],
     * ]
     * jshell&gt; array.sort(0); // sort along the first axis
     * ND: (2, 2) cpu() float32
     * [[1., 4.],
     *  [1., 3.],
     * ]
     * </pre>
     *
     * @param axis the axis to sort along
     * @return the sorted {@code NDArray}
     */
    NDArray sort(int axis);

    /**
     * Applies the softmax function along the given axis.
     *
     * @param axis the axis along which to apply
     * @return the result {@code NDArray}
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int)
     */
    NDArray softmax(int axis);

    /**
     * Applies the softmax function followed by a logarithm.
     *
     * <p>Mathematically equivalent to calling softmax and then log. This single operator is faster
     * than calling two operators and numerically more stable when computing gradients.
     *
     * @param axis the axis along which to apply
     * @return the result {@code NDArray}
     */
    NDArray logSoftmax(int axis);

    /**
     * Returns the cumulative sum of the elements in the flattened {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[1., 2., 3.],
     *  [4., 5., 6.],
     * ]
     * jshell&gt; array.cumSum(); // cumSum on flattened array
     * ND: (6) cpu() float32
     * [ 1.,  3.,  6., 10., 15., 21.]
     * </pre>
     *
     * @return the cumulative sum of the elements in the flattened {@code NDArray}
     */
    NDArray cumSum();

    /**
     * Return the cumulative sum of the elements along a given axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[1., 2., 3.],
     *  [4., 5., 6.],
     * ]
     * jshell&gt; array.cumSum(0);
     * ND: (2, 3) cpu() float32
     * [[1., 2., 3.],
     *  [5., 7., 9.],
     * ]
     * jshell&gt; array.cumSum(1);
     * ND: (2, 3) cpu() float32
     * [[ 1.,  3.,  6.],
     *  [ 4.,  9., 15.],
     * ]
     * </pre>
     *
     * @param axis the axis along which the cumulative sum is computed
     * @return the cumulative sum along the specified axis
     */
    NDArray cumSum(int axis);

    /**
     * Replace the handle of the NDArray with the other. The NDArray used for replacement will be
     * killed.
     *
     * <p>Please use with caution, this method will make the input argument unusable.
     *
     * @param replaced the handle provider that will be killed
     */
    void intern(NDArray replaced);

    /**
     * Returns the boolean {@code NDArray} with value {@code true} where this {@code NDArray}'s
     * entries are infinite, or {@code false} where they are not infinite.
     *
     * @return the boolean {@code NDArray} with value {@code true} if this {@code NDArray}'s entries
     *     are infinite
     */
    NDArray isInfinite();

    /**
     * Returns the boolean {@code NDArray} with value {@code true} where this {@code NDArray}'s
     * entries are NaN, or {@code false} where they are not NaN.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {Float.POSITIVE_INFINITY, 0, Float.NaN});
     * jshell&gt; array.isNaN();
     * ND: (3) cpu() boolean
     * [false, false,  true]
     * </pre>
     *
     * @return the boolean {@code NDArray} with value {@code true} if this {@code NDArray}'s {@link
     *     NDArray} are NaN
     */
    NDArray isNaN();

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times given
     * repeats.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(2);
     * ND: (6) cpu() float32
     * [0., 1., 2., 0., 1., 2.]
     * </pre>
     *
     * @param repeats the number of times to repeat for each dimension
     * @return a NDArray that has been tiled
     */
    NDArray tile(long repeats);

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times given by
     * repeats along given axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(1, 2);
     * ND: (1, 6) cpu() float32
     * [[0., 1., 2., 0., 1., 2.],
     * ]
     * </pre>
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return a {@code NDArray} that has been tiled
     * @throws IllegalArgumentException thrown for invalid axis
     */
    NDArray tile(int axis, long repeats);

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times given by
     * repeats.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(new long[] {2, 2});
     * ND: (2, 6) cpu() float32
     * [[0., 1., 2., 0., 1., 2.],
     *  [0., 1., 2., 0., 1., 2.],
     * ]
     * </pre>
     *
     * @param repeats the number of times to repeat along each axis
     * @return a {@code NDArray} that has been tiled
     */
    NDArray tile(long[] repeats);

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times to match
     * the desired shape.
     *
     * <p>If the desired {@link Shape}has fewer dimensions than this {@code NDArray}, it will tile
     * against the last axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(new Shape(6));
     * ND: (6) cpu() float32
     * [0., 1., 2., 0., 1., 2.]
     * </pre>
     *
     * @param desiredShape the {@link Shape}that should be converted to
     * @return a {@code NDArray} that has been tiled
     */
    NDArray tile(Shape desiredShape);

    /**
     * Repeats element of this {@code NDArray} the number of times given repeats.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.repeat(2);
     * ND: (6) cpu() float32
     * [0., 0., 1., 1., 2., 2.]
     * </pre>
     *
     * @param repeats the number of times to repeat for each axis
     * @return an {@code NDArray} that has been repeated
     */
    NDArray repeat(long repeats);

    /**
     * Repeats element of this {@code NDArray} the number of times given repeats along given axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f, 3f}, new Shape(2, 2));
     * jshell&gt; array.repeat(1, 2);
     * ND: (6) cpu() float32
     * [0., 0., 1., 1., 2., 2.]
     * </pre>
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return an {@code NDArray} that has been repeated
     * @throws IllegalArgumentException thrown for invalid axis
     */
    NDArray repeat(int axis, long repeats);

    /**
     * Repeats element of this {@code NDArray} the number of times given repeats along each axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f, 3f}, new Shape(2, 2));
     * jshell&gt; array.repeat(new long[] {2, 2});
     * ND: (12) cpu() float32
     * [0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]
     * </pre>
     *
     * @param repeats the number of times to repeat along each axis
     * @return a {@code NDArray} that has been repeated
     */
    NDArray repeat(long[] repeats);

    /**
     * Repeats element of this {@code NDArray} to match the desired shape.
     *
     * <p>If the desired {@link Shape} has fewer dimensions that the array, it will repeat against
     * the last axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f, 3f}, new Shape(2, 2));
     * jshell&gt; array.repeat(new Shape(4, 4));
     * ND: (4, 4) cpu() float32
     * [[0., 0., 1., 1.],
     *  [0., 0., 1., 1.],
     *  [2., 2., 3., 3.],
     *  [2., 2., 3., 3.],
     * ]
     * </pre>
     *
     * @param desiredShape the {@link Shape} that should be converted to
     * @return an {@code NDArray} that has been repeated
     */
    NDArray repeat(Shape desiredShape);

    /**
     * Dot product of this {@code NDArray} and the other {@code NDArray}.
     *
     * <ul>
     *   <li>If both this {@code NDArray} and the other {@code NDArray} are 1-D {@code NDArray}s, it
     *       is inner product of vectors (without complex conjugation).
     *   <li>If both this {@code NDArray} and the other {@code NDArray} are 2-D {@code NDArray}s, it
     *       is matrix multiplication.
     *   <li>If either this {@code NDArray} or the other {@code NDArray} is 0-D {@code NDArray}
     *       (scalar), it is equivalent to mul.
     *   <li>If this {@code NDArray} is N-D {@code NDArray} and the other {@code NDArray} is 1-D
     *       {@code NDArray}, it is a sum product over the last axis of those.
     *   <li>If this {@code NDArray} is N-D {@code NDArray} and the other {@code NDArray} is M-D
     *       {@code NDArray}(where M&gt;&#61;2), it is a sum product over the last axis of this
     *       {@code NDArray} and the second-to-last axis of the other {@code NDArray}
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f, 3f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {4f, 5f, 6f});
     * jshell&gt; array1.dot(array2); // inner product
     * ND: () cpu() float32
     * 32.
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {5f, 6f, 7f, 8f}, new Shape(2, 2));
     * jshell&gt; array1.dot(array2); // matrix multiplication
     * ND: (2, 2) cpu() float32
     * [[19., 22.],
     *  [43., 50.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(5f);
     * jshell&gt; array1.dot(array2);
     * ND: (2, 2) cpu() float32
     * [[ 5., 10.],
     *  [15., 20.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; array1.dot(array2);
     * ND: (2) cpu() float32
     * [ 5., 11.]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new Shape(2, 2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f, 3f ,4f}, new Shape(2, 2));
     * jshell&gt; array1.dot(array2);
     * ND: (2, 2, 2) cpu() float32
     * [[[ 7., 10.],
     *   [15., 22.],
     *  ],
     *  [[23., 34.],
     *   [31., 46.],
     *  ],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to perform dot product with
     * @return the result {@code NDArray}
     */
    NDArray dot(NDArray other);

    /**
     * Product matrix of this {@code NDArray} and the other {@code NDArray}.
     *
     * <p>The behavior depends on the arguments in the following way.
     *
     * <ul>
     *   <li>If both this {@code NDArray} and the other {@code NDArray} are 2-D {@code NDArray}s,
     *       they are multiplied like conventional matrices
     *   <li>If either this {@code NDArray} or the other {@code NDArray} is N-D {@code NDArray}, N
     *       &gt; 2 , it is treated as a stack of matrices residing in the last two indexes and
     *       broadcast accordingly.
     *   <li>If this {@code NDArray} is 1-D {@code NDArray}, it is promoted to a matrix by
     *       prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is
     *       removed.
     *   <li>If other {@code NDArray} is 1-D {@code NDArray}, it is promoted to a matrix by
     *       appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; NDArray array2 = manager.create(new float[] {4f, 1f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array1.matMul(array2); // for 2-D arrays, it is the matrix product
     * ND: (2, 2) cpu() float32
     * [[4., 1.],
     *  [2., 2.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; array1.matMul(array2);
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; array1.matMul(array2);
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array1 = manager.arange(2f * 2f * 4f).reshape(2, 2, 4);
     * jshell&gt; array2 = manager.arange(2f * 2f * 4f).reshape(2, 4, 2);
     * jshell&gt; array1.matMul(array2).get("0, 1, 1");
     * ND: () cpu() float32
     * 98.
     * </pre>
     *
     * @param other the other {@code NDArray} to perform matrix product with
     * @return the result {@code NDArray}
     */
    NDArray matMul(NDArray other);

    /**
     * Clips (limit) the values in this {@code NDArray}.
     *
     * <p>Given an interval, values outside the interval are clipped to the interval edges. For
     * example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values
     * larger than 1 become 1.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(10f);
     * jshell&gt; array.clip(1, 8);
     * ND: (10) cpu() float32
     * [1., 1., 2., 3., 4., 5., 6., 7., 8., 8.]
     * </pre>
     *
     * @param min the minimum value
     * @param max the maximum value
     * @return an {@code NDArray} with the elements of this {@code NDArray}, but where values &lt;
     *     min are replaced with min, and those &gt; max with max
     */
    NDArray clip(Number min, Number max);

    /**
     * Interchanges two axes of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f}, new Shape(1, 3));
     * jshell&gt; array;
     * ND: (1, 3) cpu() float32
     * [[1., 2., 3.],
     * ]
     * jshell&gt; array.swapAxes(0, 1);
     * ND: (3, 1) cpu() float32
     * [[1.],
     *  [2.],
     *  [3.],
     * ]
     * </pre>
     *
     * @param axis1 the first axis
     * @param axis2 the second axis
     * @return the swapped axes {@code NDArray}
     */
    default NDArray swapAxes(int axis1, int axis2) {
        int[] dims = IntStream.range(0, getShape().dimension()).toArray();
        int tmp = dims[axis1];
        dims[axis1] = dims[axis2];
        dims[axis2] = tmp;
        return transpose(dims);
    }

    /**
     * Returns the reverse order of elements in an array along the given axis.
     *
     * <p>The shape of the array is preserved, but the elements are reordered.
     *
     * @param axes the axes to flip on
     * @return the newly flipped array
     */
    NDArray flip(int... axes);

    /**
     * Returns this {@code NDArray} with axes transposed.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.transpose();
     * ND: (2, 2) cpu() float32
     * [[1., 3.],
     *  [2., 4.],
     * ]
     * </pre>
     *
     * @return the newly permuted array
     */
    NDArray transpose();

    /**
     * Returns this {@code NDArray} with given axes transposed.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.transpose(1, 0);
     * ND: (2, 2) cpu() float32
     * [[1., 3.],
     *  [2., 4.],
     * ]
     * jshell&gt; array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     *   [2., 3.],
     *  ],
     *  [[4., 5.],
     *   [6., 7.],
     *  ],
     * ]
     * jshell&gt; array.transpose(1, 0, 2);
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     *   [4., 5.],
     *  ],
     *  [[2., 3.],
     *   [6., 7.],
     *  ],
     * ]
     * </pre>
     *
     * @param axes the axes to swap to
     * @return the transposed {@code NDArray}
     * @throws IllegalArgumentException thrown when passing a axis that is greater than the actual
     *     number of dimensions
     */
    NDArray transpose(int... axes);

    /**
     * Broadcasts this {@code NDArray} to be the given shape.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.broadcast(new Shape(2, 2, 2));
     * ND: (2, 2, 2) cpu() float32
     * [[[1., 2.],
     *   [3., 4.],
     *  ],
     *  [[1., 2.],
     *   [3., 4.],
     *  ],
     * ]
     * </pre>
     *
     * @param shape the new {@link Shape} of this {@code NDArray}
     * @return the broadcasted {@code NDArray}
     */
    NDArray broadcast(Shape shape);

    /**
     * Broadcasts this {@code NDArray} to be the given shape.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     * ]
     * jshell&gt; array.broadcast(2, 2, 2);
     * ND: (2, 2, 2) cpu() float32
     * [[[1., 2.],
     *   [3., 4.],
     *  ],
     *  [[1., 2.],
     *   [3., 4.],
     *  ],
     * ]
     * </pre>
     *
     * @param shape the new {@link Shape} of this {@code NDArray}
     * @return the broadcasted {@code NDArray}
     */
    default NDArray broadcast(long... shape) {
        return broadcast(new Shape(shape));
    }

    /**
     * Returns the indices of the maximum values into the flattened {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; array.argMax();
     * ND: () cpu() int64
     * 5.
     * </pre>
     *
     * @return a {@code NDArray} containing indices
     */
    NDArray argMax();

    /**
     * Returns the indices of the maximum values along given axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; array.argMax(0);
     * ND: (3) cpu() int64
     * [1, 1, 1]
     * jshell&gt; array.argMax(1);
     * ND: (2) cpu() int64
     * [2, 2]
     * </pre>
     *
     * @param axis the axis along which to find maximum values
     * @return a {@code NDArray} containing indices
     */
    NDArray argMax(int axis);

    /**
     * Returns the indices of the minimum values into the flattened {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; array.argMin();
     * ND: () cpu() int64
     * 0.
     * </pre>
     *
     * @return a {@code NDArray} containing indices
     */
    NDArray argMin();

    /**
     * Returns the indices of the minimum values along given axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; array.argMin(0);
     * ND: (3) cpu() int64
     * [0, 0, 0]
     * jshell&gt; array.argMin(1);
     * ND: (2) cpu() int64
     * [0, 0]
     * </pre>
     *
     * @param axis the axis along which to find minimum values
     * @return a {@code NDArray} containing indices
     */
    NDArray argMin(int axis);

    /**
     * Returns percentile for this {@code NDArray}.
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
     * Returns median value for this {@code NDArray}.
     *
     * @return the median {@code NDArray}
     */
    NDArray median();

    /**
     * Returns median value along given axes.
     *
     * @param axes the axes along which to perform the median operation
     * @return the median {@code NDArray} along the specified axes
     */
    NDArray median(int[] axes);

    // ------------ Sparse methods ------------

    /**
     * Returns a dense representation of the sparse {@code NDArray}.
     *
     * @return the result {@code NDArray}
     */
    NDArray toDense();

    /**
     * Returns a sparse representation of {@code NDArray}.
     *
     * @param fmt the {@link SparseFormat} of this {@code NDArray}
     * @return the result {@code NDArray}
     */
    NDArray toSparse(SparseFormat fmt);

    /**
     * Returns the indices of elements that are non-zero.
     *
     * <p>Note that the behavior is slightly different from numpy.nonzero. Numpy returns a tuple of
     * NDArray, one for each dimension of NDArray. DJL nonzero returns only one {@code NDArray} with
     * last dimension containing all dimension of indices.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 1f, 1f, 0f, 1f});
     * jshell&gt; array.nonzero();
     * ND: (4, 1) cpu() int64
     * [[ 0],
     *  [ 1],
     *  [ 2],
     *  [ 4],
     * ]
     * jshell&gt; array = manager.create(new float[] {3f, 0f, 0f, 0f, 4f, 0f, 5f, 6f, 0f}).reshape(3, 3);
     * jshell&gt; array;
     * ND: (3, 3) cpu() float32
     * [[3., 0., 0.],
     *  [0., 4., 0.],
     *  [5., 6., 0.],
     * ]
     * jshell&gt; array.nonzero();
     * ND: (4, 2) cpu() int64
     * [[ 0,  0],
     *  [ 1,  1],
     *  [ 2,  0],
     *  [ 2,  1],
     * ]
     * </pre>
     *
     * @return the indices of the elements that are non-zero
     */
    NDArray nonzero();

    /**
     * Returns {@code true} if this {@code NDArray} is special case: no-value {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new Shape(2, 0, 1));
     * jshell&gt; array;
     * ND: (2, 0, 1) cpu() float32
     * []
     * jshell&gt; array.isEmpty();
     * true
     * </pre>
     *
     * @return {@code true} if this NDArray is empty
     */
    default boolean isEmpty() {
        return getShape().size() == 0;
    }

    /**
     * Returns {@code true} if all elements within this {@code NDArray} are non-zero or {@code
     * true}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true, false, true, true}, new Shape(2, 2));
     * jshell&gt; array.all();
     * ND: () cpu() boolean
     * false
     * jshell&gt; NDArray array = manager.create(new float[] {-1f, 4f, 5f});
     * jshell&gt; array.all(); // all elements are non-zero
     * ND: () cpu() boolean
     * true
     * </pre>
     *
     * @return {@code true} if all elements within this {@code NDArray} are non-zero or {@code true}
     */
    default NDArray all() {
        // result of sum operation is int64 now
        return toType(DataType.BOOLEAN, false).sum().eq(size());
    }

    /**
     * Returns {@code true} if any of the elements within this {@code NDArray} are non-zero or
     * {@code true}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true, false, true, true}, new Shape(2, 2));
     * jshell&gt; array.any();
     * ND: () cpu() boolean
     *  true
     * jshell&gt; NDArray array = manager.create(new float[] {-1, 0, 5});
     * jshell&gt; array.any() // all elements are non-zero
     * ND: () cpu() boolean
     *  true
     * </pre>
     *
     * @return {@code true} if any of the elements within this {@code NDArray} are non-zero or
     *     {@code true}
     */
    default NDArray any() {
        return toType(DataType.BOOLEAN, false).sum().gt(0);
    }

    /**
     * Returns {@code true} if none of the elements within this {@code NDArray} are non-zero or
     * {@code true}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {false, false});
     * jshell&gt; array.none();
     * ND: () cpu() boolean
     *  true
     * jshell&gt; NDArray array = manager.create(new float[] {-1f, 0f, 5f});
     * jshell&gt; array.none() // all elements are non-zero
     * ND: () cpu() boolean
     * false
     * </pre>
     *
     * @return {@code true} if none of the elements within this {@code NDArray} are non-zero or
     *     {@code true}
     */
    default NDArray none() {
        return toType(DataType.BOOLEAN, false).sum().eq(0);
    }

    /**
     * Counts the number of non-zero values in this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 0f, 1f, 2f, 7f, 0f}, new Shape(2, 3));
     * jshell&gt; array.countNonzero()
     * ND: () cpu() int64
     * 3
     * </pre>
     *
     * @return the number of non-zero values in this {@code NDArray}
     */
    default NDArray countNonzero() {
        return toType(DataType.BOOLEAN, false).sum();
    }

    /**
     * Counts the number of non-zero values in this {@code NDArray} along a given axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 0f, 1f, 2f, 7f, 0f}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 0., 1.],
     *  [2., 7., 0.],
     * ]
     * jshell&gt; array.countNonzero(0);
     * ND: (3) cpu() int64
     * [ 1,  1,  1]
     * jshell&gt; array.countNonzero(1);
     * ND: (2) cpu() int64
     * [ 1,  2]
     * </pre>
     *
     * @param axis the axis to operate on
     * @return the number of non-zero values in this {@code NDArray} along a given axis
     */
    default NDArray countNonzero(int axis) {
        return toType(DataType.BOOLEAN, false).sum(new int[] {axis});
    }

    /**
     * Returns element-wise inverse gauss error function of the {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 0.5f, -1f});
     * jshell&gt; array.erfinv();
     * ND: (3) cpu() float32
     * [0., 0.4769, -inf]
     * </pre>
     *
     * @return The inverse of gauss error of the {@code NDArray}, element-wise
     */
    NDArray erfinv();

    /**
     * Returns an internal representative of Native {@code NDArray}.
     *
     * <p>This method should only be used by Engine provider
     *
     * @return an internal representative of Native {@code NDArray}
     */
    NDArrayEx getNDArrayInternal();

    /**
     * Runs the debug string representation of this {@code NDArray}.
     *
     * @param maxSize the maximum elements to print out
     * @param maxDepth the maximum depth to print out
     * @param maxRows the maximum rows to print out
     * @param maxColumns the maximum columns to print out
     * @return the debug string representation of this {@code NDArray}
     */
    default String toDebugString(int maxSize, int maxDepth, int maxRows, int maxColumns) {
        return NDFormat.format(this, maxSize, maxDepth, maxRows, maxColumns);
    }

    /** {@inheritDoc} */
    @Override
    void close();

    /**
     * Returns the norm of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-3f, -4f});
     * jshell&gt; array.norm();
     * ND: () cpu() float32
     * 5.
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm();
     * ND: () cpu() float32
     * 5.4472
     * </pre>
     *
     * @return the norm of this {@code NDArray}
     */
    default NDArray norm() {
        return norm(false);
    }

    /**
     * Returns the norm of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-3f, -4f});
     * jshell&gt; array.norm(new int[] {0});
     * ND: () cpu() float32
     * 5.
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(new int[] {0});
     * ND: (2) cpu() float32
     * [3.1623, 4.4721]
     * </pre>
     *
     * @param axes If axes contains an integer, it specifies the axis of x along which to compute
     *     the vector norms. If axis contains 2 integers, it specifies the axes that hold 2-D
     *     matrices, and the matrix norms of these matrices are computed.
     * @return the norm of this {@code NDArray}
     */
    default NDArray norm(int[] axes) {
        return norm(axes, false);
    }

    /**
     * Returns the norm of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-3f, -4f});
     * jshell&gt; array.norm(true);
     * ND: () cpu() float32
     * 5.
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(true);
     * ND: () cpu() float32
     * [[5.4772],
     * ]
     * </pre>
     *
     * @param keepDims If this is set to True, the axes which are normed over are left in the result
     *     as dimensions with size one. With this option the result will broadcast correctly against
     *     the original x.
     * @return the norm of this {@code NDArray}
     */
    NDArray norm(boolean keepDims);

    /**
     * Returns the norm of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[3.1623, 4.4721],
     * ]
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(new int[] {0}, false);
     * ND: (2) cpu() float32
     * [3.1623, 4.4721]
     * </pre>
     *
     * @param axes If axes contains an integer, it specifies the axis of x along which to compute
     *     the vector norms. If axis contains 2 integers, it specifies the axes that hold 2-D
     *     matrices, and the matrix norms of these matrices are computed.
     * @param keepDims keepDims If this is set to True, the axes which are normed over are left in
     *     the result as dimensions with size one. With this option the result will broadcast
     *     correctly against the original x.
     * @return the norm of this {@code NDArray}
     */
    default NDArray norm(int[] axes, boolean keepDims) {
        return norm(2, axes, keepDims);
    }

    /**
     * Returns the norm of this {@code NDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(2, new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[3.1623, 4.4721],
     * ]
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(2, new int[] {0}, false);
     * ND: (2) cpu() float32
     * [3.1623, 4.4721]
     * </pre>
     *
     * @param ord Order of the norm.
     * @param axes If axes contains an integer, it specifies the axis of x along which to compute
     *     the vector norms. If axis contains 2 integers, it specifies the axes that hold 2-D
     *     matrices, and the matrix norms of these matrices are computed.
     * @param keepDims keepDims If this is set to True, the axes which are normed over are left in
     *     the result as dimensions with size one. With this option the result will broadcast
     *     correctly against the original x.
     * @return the norm of this {@code NDArray}
     */
    NDArray norm(int ord, int[] axes, boolean keepDims);

    /**
     * Returns a one-hot {@code NDArray}.
     *
     * <ul>
     *   <li>The locations represented by indices take value 1, while all other locations take value
     *       0.
     *   <li>If the input {@code NDArray} is rank N, the output will have rank N+1. The new axis is
     *       appended at the end.
     *   <li>If {@code NDArray} is a scalar the output shape will be a vector of length depth.
     *   <li>If {@code NDArray} is a vector of length features, the output shape will be features x
     *       depth.
     *   <li>If {@code NDArray} is a matrix with shape [batch, features], the output shape will be
     *       batch x features x depth.
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new int[] {1, 0, 2, 0});
     * jshell&gt; array.oneHot(3);
     * ND: (4, 3) cpu() float32
     * [[0., 1., 0.],
     *  [1., 0., 0.],
     *  [0., 0., 1.],
     *  [1., 0., 0.],
     * ]
     * jshell&gt; NDArray array = manager.create(new int[][] {{1, 0}, {1, 0}, {2, 0}});
     * jshell&gt; array.oneHot(3);
     * ND: (3, 2, 3) cpu() float32
     * [[[0., 1., 0.],
     *   [1., 0., 0.],
     *  ],
     *  [[0., 1., 0.],
     *   [1., 0., 0.],
     *  ],
     *  [[0., 0., 1.],
     *   [1., 0., 0.],
     *  ],
     * ]
     * </pre>
     *
     * @param depth Depth of the one hot dimension.
     * @return one-hot encoding of this {@code NDArray}
     * @see <a
     *     href=https://d2l.djl.ai/chapter_linear-networks/softmax-regression.html#classification-problems>Classification-problems</a>
     */
    default NDArray oneHot(int depth) {
        return oneHot(depth, 1f, 0f, DataType.FLOAT32);
    }

    /**
     * Returns a one-hot {@code NDArray}.
     *
     * <ul>
     *   <li>The locations represented by indices take value 1, while all other locations take value
     *       0.
     *   <li>If the input {@code NDArray} is rank N, the output will have rank N+1. The new axis is
     *       appended at the end.
     *   <li>If {@code NDArray} is a scalar the output shape will be a vector of length depth.
     *   <li>If {@code NDArray} is a vector of length features, the output shape will be features x
     *       depth.
     *   <li>If {@code NDArray} is a matrix with shape [batch, features], the output shape will be
     *       batch x features x depth.
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new int[] {1, 0, 2, 0});
     * jshell&gt; array.oneHot(3);
     * ND: (4, 3) cpu() float32
     * [[0., 1., 0.],
     *  [1., 0., 0.],
     *  [0., 0., 1.],
     *  [1., 0., 0.],
     * ]
     * jshell&gt; NDArray array = manager.create(new int[][] {{1, 0}, {1, 0}, {2, 0}});
     * jshell&gt; array.oneHot(3);
     * ND: (3, 2, 3) cpu() float32
     * [[[0., 1., 0.],
     *   [1., 0., 0.],
     *  ],
     *  [[0., 1., 0.],
     *   [1., 0., 0.],
     *  ],
     *  [[0., 0., 1.],
     *   [1., 0., 0.],
     *  ],
     * ]
     * </pre>
     *
     * @param depth Depth of the one hot dimension.
     * @param dataType dataType of the output.
     * @return one-hot encoding of this {@code NDArray}
     * @see <a
     *     href=https://d2l.djl.ai/chapter_linear-networks/softmax-regression.html#classification-problems>Classification-problems</a>
     */
    default NDArray oneHot(int depth, DataType dataType) {
        return oneHot(depth, 0f, 1f, dataType);
    }

    /**
     * Returns a one-hot {@code NDArray}.
     *
     * <ul>
     *   <li>The locations represented by indices take value onValue, while all other locations take
     *       value offValue.
     *   <li>If the input {@code NDArray} is rank N, the output will have rank N+1. The new axis is
     *       appended at the end.
     *   <li>If {@code NDArray} is a scalar the output shape will be a vector of length depth.
     *   <li>If {@code NDArray} is a vector of length features, the output shape will be features x
     *       depth.
     *   <li>If {@code NDArray} is a matrix with shape [batch, features], the output shape will be
     *       batch x features x depth.
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new int[] {1, 0, 2, 0});
     * jshell&gt; array.oneHot(3, 8f, 1f, array.getDataType());
     * ND: (4, 3) cpu() int32
     * [[ 1,  8,  1],
     *  [ 8,  1,  1],
     *  [ 1,  1,  8],
     *  [ 8,  1,  1],
     * ]
     * </pre>
     *
     * @param depth Depth of the one hot dimension.
     * @param onValue The value assigned to the locations represented by indices.
     * @param offValue The value assigned to the locations not represented by indices.
     * @param dataType dataType of the output.
     * @return one-hot encoding of this {@code NDArray}
     * @see <a
     *     href=https://d2l.djl.ai/chapter_linear-networks/softmax-regression.html#classification-problems>Classification-problems</a>
     */
    NDArray oneHot(int depth, float onValue, float offValue, DataType dataType);

    /**
     * Batchwise product of this {@code NDArray} and the other {@code NDArray}.
     *
     * <ul>
     *   <li>batchDot is used to compute dot product of x and y when x and y are data in batch,
     *       namely N-D (N greater or equal to 3) arrays in shape of (B0, …, B_i, :, :). For
     *       example, given x with shape (B_0, …, B_i, N, M) and y with shape (B_0, …, B_i, M, K),
     *       the result array will have shape (B_0, …, B_i, N, K), which is computed by:
     *       batch_dot(x,y)[b_0, ..., b_i, :, :] = dot(x[b_0, ..., b_i, :, :], y[b_0, ..., b_i, :,
     *       :])
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.ones(new Shape(2, 1, 4));
     * jshell&gt; NDArray array2 = manager.ones(new Shape(2, 4, 6));
     * jshell&gt; array1.batchDot(array2);
     * ND: (2, 1, 6) cpu() float32
     * [[[4., 4., 4., 4., 4., 4.],
     *  ],
     *  [[4., 4., 4., 4., 4., 4.],
     *  ],
     * ]
     * </pre>
     *
     * @param other the other {@code NDArray} to perform batch dot product with
     * @return the result {@code NDArray}
     */
    NDArray batchDot(NDArray other);
}
