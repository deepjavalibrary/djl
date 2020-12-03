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
package ai.djl.paddlepaddle.engine;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.paddlepaddle.jna.JnaUtils;
import ai.djl.util.NativeResource;
import com.sun.jna.Pointer;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Arrays;

/** {@code PpNDArray} is the PaddlePaddle implementation of {@link NDArray}. */
public class PpNDArray extends NativeResource<Pointer> implements NDArray {

    private PpNDManager manager;
    private Shape shape;
    private DataType dataType;

    /**
     * Constructs an PpNDArray from a native handle (internal. Use {@link NDManager} instead).
     *
     * @param manager the manager to attach the new array to
     * @param handle the pointer to the native MxNDArray memory
     */
    public PpNDArray(PpNDManager manager, Pointer handle) {
        super(handle);
        this.manager = manager;
        manager.attach(getUid(), this);
    }

    /**
     * Constructs an PaddlePaddle NDArray from a {@link PpNDManager} (internal. Use {@link
     * NDManager} instead).
     *
     * @param manager the manager to attach the new array to
     * @param pointer the native tensor handle
     * @param shape the shape of {@code PpNDArray}
     * @param dataType the data type of {@code PpNDArray}
     */
    public PpNDArray(PpNDManager manager, Pointer pointer, Shape shape, DataType dataType) {
        super(pointer);
        this.manager = manager;
        this.shape = shape;
        this.dataType = dataType;
        manager.attach(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return JnaUtils.getNdArrayName(this);
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        JnaUtils.setNdArrayName(this, name);
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (dataType == null) {
            dataType = JnaUtils.getDataType(this);
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = JnaUtils.getShape(this);
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDManager attach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (PpNDManager) manager;
        manager.attach(getUid(), this);
        return original;
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detach(getUid());
        manager = PpNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient(SparseFormat sparseFormat) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasGradient() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return JnaUtils.getByteBufferFromNd(this);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sign() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray signi() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flip(int... axes) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... axes) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] axes) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray erfinv() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isReleased()) {
            return "This array is already closed";
        }
        return "ND: "
                + getShape()
                + ' '
                + getDevice()
                + ' '
                + getDataType()
                + '\n'
                + Arrays.toString(toArray());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeNdArray(pointer);
        }
    }
}
