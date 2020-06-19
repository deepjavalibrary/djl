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
package ai.djl.onnxruntime.engine;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.UUID;

/** {@code OrtNDArray} is the ONNX Runtime implementation of {@link NDArray}. */
public class OrtNDArray implements NDArray {

    private OrtNDManager manager;
    private OnnxTensor tensor;
    private Shape shape;
    private DataType dataType;
    private String name;
    private boolean isClosed;
    private String uid;

    /**
     * Constructs an ONNX Runtime NDArray from a {@link OnnxTensor} (internal. Use {@link NDManager}
     * instead).
     *
     * @param manager the manager to attach the new array to
     * @param tensor the {@link OnnxTensor} to the ONNX Runtime
     */
    OrtNDArray(OrtNDManager manager, OnnxTensor tensor) {
        this.manager = manager;
        this.tensor = tensor;
        uid = UUID.randomUUID().toString();
        manager.attach(uid, this);
    }

    OnnxTensor getTensor() {
        return tensor;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
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
    public String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (dataType != null) {
            return dataType;
        }
        OnnxJavaType javaType = tensor.getInfo().type;
        switch (javaType) {
            case FLOAT:
                dataType = DataType.FLOAT32;
                break;
            case DOUBLE:
                dataType = DataType.FLOAT64;
                break;
            case INT8:
                dataType = DataType.INT8;
                break;
            case INT32:
                dataType = DataType.INT32;
                break;
            case INT64:
                dataType = DataType.INT64;
                break;
            case BOOL:
                dataType = DataType.BOOLEAN;
                break;
            case UNKNOWN:
                dataType = DataType.UNKNOWN;
                break;
            default:
                throw new UnsupportedOperationException("type is not supported: " + javaType);
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        // TODO: Support on multiple devices
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape != null) {
            return shape;
        }
        shape = new Shape(tensor.getInfo().getShape());
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (OrtNDManager) manager;
        manager.attach(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detach(getUid());
        manager = OrtNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient(SparseFormat sparseFormat) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasGradient() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        ByteBuffer bb = getTensor().getByteBuffer();
        bb.order(ByteOrder.nativeOrder());
        return bb;
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, NDArray value) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Number value) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void setScalar(NDIndex index, Number value) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... axes) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] axes) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isClosed) {
            return "This array is already closed";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("ND: ")
                .append(getShape())
                .append(' ')
                .append(getDevice())
                .append(' ')
                .append(getDataType())
                .append('\n')
                .append(Arrays.toString(toArray()));
        return sb.toString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        tensor.close();
        isClosed = true;
    }
}
