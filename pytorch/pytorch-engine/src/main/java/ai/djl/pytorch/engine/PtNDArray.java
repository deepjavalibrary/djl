/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.ndarray.Matrix;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.index.NDIndexFullSlice;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.internal.NDFormat;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.pytorch.jni.NativeResource;
import ai.djl.pytorch.jni.Pointer;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.function.Predicate;

/** {@code PtNDArray} is the PyTorch implementation of {@link NDArray}. */
public class PtNDArray extends NativeResource implements NDArray {

    private static final int MAX_SIZE = 100;
    private static final int MAX_DEPTH = 10;
    private static final int MAX_ROWS = 10;
    private static final int MAX_COLUMNS = 20;

    private String name;
    private Device device;
    private DataType dataType;
    private Shape shape;
    private SparseFormat sparseFormat;
    private PtNDManager manager;
    private PtNDArrayEx ptNDArrayEx;

    /**
     * Constructs an PtTorch from a native handle and metadata (internal. Use {@link NDManager}
     * instead).
     *
     * @param manager the manager to attach the new array to
     * @param handle the pointer to the native PtTorch memory
     * @param device the device the new array will be located on
     * @param shape the shape of the new array
     * @param dataType the dataType of the new array
     */
    PtNDArray(PtNDManager manager, Pointer handle, Device device, Shape shape, DataType dataType) {
        this(manager, handle);
        this.device = device;
        // shape check
        if (Arrays.stream(shape.getShape()).anyMatch(s -> s < 0)) {
            throw new IllegalArgumentException("The shape must be >= 0");
        }
        this.shape = shape;
        this.dataType = dataType;
    }

    /**
     * Constructs an PyTorch from a native handle (internal. Use {@link NDManager} instead).
     *
     * @param manager the manager to attach the new array to
     * @param handle the pointer to the native PyTorch memory
     */
    public PtNDArray(PtNDManager manager, Pointer handle) {
        super(handle);
        this.manager = manager;
        this.ptNDArrayEx = new PtNDArrayEx(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (dataType == null) {
            dataType = JniUtils.getDataType(this);
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        if (device == null) {
            device = JniUtils.getDevice(this);
        }
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = JniUtils.getShape(this);
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        if (sparseFormat == null) {
            sparseFormat = JniUtils.getSparseFormat(this);
        }
        return sparseFormat;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        return JniUtils.to(this, getDataType(), device, copy);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        return JniUtils.to(this, dataType, getDevice(), copy);
    }

    /** {@inheritDoc} */
    @Override
    public Matrix toMatrix() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return JniUtils.getByteBuffer(this);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {}

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, NDArray value) {}

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Number value) {}

    /** {@inheritDoc} */
    @Override
    public void setScalar(NDIndex index, Number value) {}

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        // TODO add full support of index and refactor NDIndex
        NDIndexFullSlice fullSlice = index.getAsFullSlice(getShape()).orElse(null);
        if (isScalar()) {
            return this;
        }
        return JniUtils.get(this, 0, fullSlice.getMin()[0]);
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {}

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        return JniUtils.contentEqual(this, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return JniUtils.eq(this, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return JniUtils.neq(this, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return JniUtils.gt(this, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return JniUtils.gte(this, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return JniUtils.lt(this, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return JniUtils.lte(this, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return JniUtils.sub(this, n.doubleValue());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return JniUtils.div(this, n.doubleValue());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray others) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return JniUtils.abs(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return JniUtils.sqrt(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return JniUtils.floor(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return JniUtils.ceil(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return JniUtils.round(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return JniUtils.trunc(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return JniUtils.exp(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return JniUtils.log(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return JniUtils.log10(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return JniUtils.log2(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return JniUtils.sin(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return JniUtils.cos(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return JniUtils.tan(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return JniUtils.asin(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return JniUtils.acos(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return JniUtils.atan(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return JniUtils.sinh(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return JniUtils.cosh(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return JniUtils.tanh(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int sections, int axis) {
        return JniUtils.split(this, sections, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int[] indices, int axis) {
        long[] longIndices = Arrays.stream(indices).asLongStream().toArray();
        return JniUtils.split(this, longIndices, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return JniUtils.reshape(this, shape.getShape());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        if (temperature != 1.0) {
            throw new UnsupportedOperationException("PyTorch softmax didn't suuport temperature");
        }
        return JniUtils.softmax(this, axes[0], getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int[] axes, double temperature) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(NDIndex index) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        return JniUtils.argMax(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        return JniUtils.argMax(this, axis, false);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return ptNDArrayEx;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return toDebugString(MAX_SIZE, MAX_DEPTH, MAX_ROWS, MAX_COLUMNS);
    }

    /**
     * Runs the debug string representation of this {@code NDArray}.
     *
     * @param maxSize the maximum elements to print out
     * @param maxDepth the maximum depth to print out
     * @param maxRows the maximum rows to print out
     * @param maxColumns the maximum columns to print out
     * @return the debug string representation of this {@code NDArray}
     */
    public String toDebugString(int maxSize, int maxDepth, int maxRows, int maxColumns) {
        if (isReleased()) {
            return "This array is already closed";
        }
        return NDFormat.format(this, maxSize, maxDepth, maxRows, maxColumns);
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof PtNDArray) {
            return contentEquals((PtNDArray) obj);
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteNdArray(pointer);
            manager.detach(getUid());
            manager = null;
        }
    }
}
