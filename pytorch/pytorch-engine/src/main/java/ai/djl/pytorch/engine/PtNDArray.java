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
        return null;
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
        return null;
    }

    @Override
    public NDArray toDevice(Device device, boolean copy) {
        return null;
    }

    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        return null;
    }

    @Override
    public Matrix toMatrix() {
        return null;
    }

    @Override
    public void attachGradient() {}

    @Override
    public NDArray getGradient() {
        return null;
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
        return null;
    }

    @Override
    public void copyTo(NDArray array) {}

    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        return null;
    }

    @Override
    public NDArray zerosLike() {
        return null;
    }

    @Override
    public NDArray onesLike() {
        return null;
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
        return null;
    }

    @Override
    public NDArray eq(NDArray other) {
        return null;
    }

    @Override
    public NDArray neq(Number other) {
        return null;
    }

    @Override
    public NDArray neq(NDArray other) {
        return null;
    }

    @Override
    public NDArray gt(Number other) {
        return null;
    }

    @Override
    public NDArray gt(NDArray other) {
        return null;
    }

    @Override
    public NDArray gte(Number other) {
        return null;
    }

    @Override
    public NDArray gte(NDArray other) {
        return null;
    }

    @Override
    public NDArray lt(Number other) {
        return null;
    }

    @Override
    public NDArray lt(NDArray other) {
        return null;
    }

    @Override
    public NDArray lte(Number other) {
        return null;
    }

    @Override
    public NDArray lte(NDArray other) {
        return null;
    }

    @Override
    public NDArray add(Number n) {
        return null;
    }

    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    @Override
    public NDArray sub(Number n) {
        return null;
    }

    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    @Override
    public NDArray mul(Number n) {
        return null;
    }

    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    @Override
    public NDArray div(Number n) {
        return null;
    }

    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    @Override
    public NDArray mod(Number n) {
        return null;
    }

    @Override
    public NDArray mod(NDArray other) {
        return null;
    }

    @Override
    public NDArray pow(Number n) {
        return null;
    }

    @Override
    public NDArray pow(NDArray other) {
        return null;
    }

    @Override
    public NDArray addi(Number n) {
        return null;
    }

    @Override
    public NDArray addi(NDArray other) {
        return null;
    }

    @Override
    public NDArray subi(Number n) {
        return null;
    }

    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    @Override
    public NDArray muli(Number n) {
        return null;
    }

    @Override
    public NDArray muli(NDArray others) {
        return null;
    }

    @Override
    public NDArray divi(Number n) {
        return null;
    }

    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    @Override
    public NDArray modi(Number n) {
        return null;
    }

    @Override
    public NDArray modi(NDArray other) {
        return null;
    }

    @Override
    public NDArray powi(Number n) {
        return null;
    }

    @Override
    public NDArray powi(NDArray other) {
        return null;
    }

    @Override
    public NDArray maximum(Number n) {
        return null;
    }

    @Override
    public NDArray maximum(NDArray other) {
        return null;
    }

    @Override
    public NDArray minimum(Number n) {
        return null;
    }

    @Override
    public NDArray minimum(NDArray other) {
        return null;
    }

    @Override
    public NDArray neg() {
        return null;
    }

    @Override
    public NDArray negi() {
        return null;
    }

    @Override
    public NDArray abs() {
        return null;
    }

    @Override
    public NDArray square() {
        return null;
    }

    @Override
    public NDArray cbrt() {
        return null;
    }

    @Override
    public NDArray floor() {
        return null;
    }

    @Override
    public NDArray ceil() {
        return null;
    }

    @Override
    public NDArray round() {
        return null;
    }

    @Override
    public NDArray trunc() {
        return null;
    }

    @Override
    public NDArray exp() {
        return null;
    }

    @Override
    public NDArray log() {
        return null;
    }

    @Override
    public NDArray log10() {
        return null;
    }

    @Override
    public NDArray log2() {
        return null;
    }

    @Override
    public NDArray sin() {
        return null;
    }

    @Override
    public NDArray cos() {
        return null;
    }

    @Override
    public NDArray tan() {
        return null;
    }

    @Override
    public NDArray asin() {
        return null;
    }

    @Override
    public NDArray acos() {
        return null;
    }

    @Override
    public NDArray atan() {
        return null;
    }

    @Override
    public NDArray sinh() {
        return null;
    }

    @Override
    public NDArray cosh() {
        return null;
    }

    @Override
    public NDArray tanh() {
        return null;
    }

    @Override
    public NDArray asinh() {
        return null;
    }

    @Override
    public NDArray acosh() {
        return null;
    }

    @Override
    public NDArray atanh() {
        return null;
    }

    @Override
    public NDArray toDegrees() {
        return null;
    }

    @Override
    public NDArray toRadians() {
        return null;
    }

    @Override
    public NDArray max() {
        return null;
    }

    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray min() {
        return null;
    }

    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray sum() {
        return null;
    }

    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray prod() {
        return null;
    }

    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray mean() {
        return null;
    }

    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return null;
    }

    @Override
    public NDList split(int[] indices, int axis) {
        return null;
    }

    @Override
    public NDArray flatten() {
        return null;
    }

    @Override
    public NDArray reshape(Shape shape) {
        return null;
    }

    @Override
    public NDArray expandDims(int axis) {
        return null;
    }

    @Override
    public NDArray squeeze(int[] axes) {
        return null;
    }

    @Override
    public NDArray logicalAnd(NDArray other) {
        return null;
    }

    @Override
    public NDArray logicalOr(NDArray other) {
        return null;
    }

    @Override
    public NDArray logicalXor(NDArray other) {
        return null;
    }

    @Override
    public NDArray logicalNot() {
        return null;
    }

    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return null;
    }

    @Override
    public NDArray sort() {
        return null;
    }

    @Override
    public NDArray sort(int axis) {
        return null;
    }

    @Override
    public NDArray softmax(int[] axes, double temperature) {
        return null;
    }

    @Override
    public NDArray cumSum() {
        return null;
    }

    @Override
    public NDArray cumSum(int axis) {
        return null;
    }

    @Override
    public NDArray isInfinite() {
        return null;
    }

    @Override
    public NDArray isNaN() {
        return null;
    }

    @Override
    public NDArray createMask(NDIndex index) {
        return null;
    }

    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        return null;
    }

    @Override
    public NDArray tile(long repeats) {
        return null;
    }

    @Override
    public NDArray tile(int axis, long repeats) {
        return null;
    }

    @Override
    public NDArray tile(long[] repeats) {
        return null;
    }

    @Override
    public NDArray tile(Shape desiredShape) {
        return null;
    }

    @Override
    public NDArray repeat(long repeats) {
        return null;
    }

    @Override
    public NDArray repeat(int axis, long repeats) {
        return null;
    }

    @Override
    public NDArray repeat(long[] repeats) {
        return null;
    }

    @Override
    public NDArray repeat(Shape desiredShape) {
        return null;
    }

    @Override
    public NDArray dot(NDArray other) {
        return null;
    }

    @Override
    public NDArray clip(Number min, Number max) {
        return null;
    }

    @Override
    public NDArray transpose() {
        return null;
    }

    @Override
    public NDArray transpose(int... axes) {
        return null;
    }

    @Override
    public NDArray broadcast(Shape shape) {
        return null;
    }

    @Override
    public NDArray argMax() {
        return null;
    }

    @Override
    public NDArray argMax(int axis) {
        return null;
    }

    @Override
    public NDArray argMin() {
        return null;
    }

    @Override
    public NDArray argMin(int axis) {
        return null;
    }

    @Override
    public NDArray percentile(Number percentile) {
        return null;
    }

    @Override
    public NDArray percentile(Number percentile, int[] axes) {
        return null;
    }

    @Override
    public NDArray median() {
        return null;
    }

    @Override
    public NDArray median(int[] axes) {
        return null;
    }

    @Override
    public NDArray toDense() {
        return null;
    }

    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return null;
    }

    @Override
    public NDArray nonzero() {
        return null;
    }

    @Override
    public NDArrayEx getNDArrayInternal() {
        return null;
    }

    @Override
    public void close() {
        // TODO: Implement close method
    }
}
