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
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.ndarray.Matrix;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
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

public class PtNDArray extends NativeResource implements NDArray {

    private Device device;
    private DataType dataType;
    private Shape shape;
    private PtNDManager manager;

    public PtNDArray(
            PtNDManager manager, Pointer handle, Device device, Shape shape, DataType dataType) {
        this(manager, handle);
        this.device = device;
        // shape check
        if (Arrays.stream(shape.getShape()).anyMatch(s -> s < 0)) {
            throw new IllegalArgumentException("The shape must be >= 0");
        }
        this.shape = shape;
        this.dataType = dataType;
    }

    public PtNDArray(PtNDManager manager, Pointer handle) {
        super(handle);
        this.manager = manager;
    }

    @Override
    public NDManager getManager() {
        return manager;
    }

    @Override
    public String getName() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void setName(String name) {}

    @Override
    public DataType getDataType() {
        if (dataType == null) {
            dataType = JniUtils.getDataType(this);
        }
        return dataType;
    }

    @Override
    public Device getDevice() {
        if (device == null) {
            device = JniUtils.getDevice(this);
        }
        return device;
    }

    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = JniUtils.getShape(this);
        }
        return shape;
    }

    @Override
    public SparseFormat getSparseFormat() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray toDevice(Device device, boolean copy) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public Matrix toMatrix() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void attachGradient() {}

    @Override
    public NDArray getGradient() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public ByteBuffer toByteBuffer() {
        return JniUtils.getByteBuffer(this);
    }

    @Override
    public void set(Buffer data) {}

    @Override
    public void set(NDIndex index, NDArray value) {}

    @Override
    public void set(NDIndex index, Number value) {}

    @Override
    public void setScalar(NDIndex index, Number value) {}

    @Override
    public NDArray get(NDIndex index) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void copyTo(NDArray array) {}

    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray zerosLike() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray onesLike() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    @Override
    public boolean contentEquals(NDArray other) {
        return false;
    }

    @Override
    public NDArray eq(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray eq(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray neq(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray neq(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray gt(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray gt(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray gte(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray gte(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray lt(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray lt(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray lte(Number other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray lte(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray add(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray add(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sub(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sub(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray mul(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray mul(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray div(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray div(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray mod(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray mod(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray pow(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray pow(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray addi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray addi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray subi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray subi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray muli(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray muli(NDArray others) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray divi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray divi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray modi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray modi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray powi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray powi(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray maximum(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray maximum(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray minimum(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray minimum(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray neg() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray negi() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray abs() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray square() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray cbrt() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray floor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray ceil() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray round() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray trunc() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray exp() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray log() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray log10() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray log2() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sin() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray cos() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray tan() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray asin() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray acos() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray atan() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sinh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray cosh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray tanh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray asinh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray acosh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray atanh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray toDegrees() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray toRadians() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray max() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray min() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sum() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray prod() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray mean() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDList split(int[] indices, int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray flatten() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray reshape(Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray expandDims(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray squeeze(int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray logicalAnd(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray logicalOr(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray logicalXor(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray logicalNot() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray argSort(int axis, boolean ascending) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sort() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray sort(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray softmax(int[] axes, double temperature) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray logSoftmax(int[] axes, double temperature) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray cumSum() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray cumSum(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray isInfinite() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray isNaN() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray createMask(NDIndex index) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray tile(long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray tile(int axis, long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray tile(long[] repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray tile(Shape desiredShape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray repeat(long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray repeat(int axis, long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray repeat(long[] repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray repeat(Shape desiredShape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray dot(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray clip(Number min, Number max) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray transpose() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray transpose(int... axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray broadcast(Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray argMax() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray argMax(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray argMin() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray argMin(int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray percentile(Number percentile, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray median() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray toDense() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArray nonzero() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NDArrayEx getNDArrayInternal() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void close() {
        JniUtils.deleteNdArray(this);
    }
}
