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
package ai.djl.test.mock;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.index.NDIndexFixed;
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
import java.util.UUID;
import java.util.function.Predicate;

public class MockNDArray implements NDArray {

    private String name;
    private Device device;
    private String uid;
    private SparseFormat sparseFormat;
    private DataType dataType;
    private Shape shape;
    private NDManager manager;
    private ByteBuffer data;

    public MockNDArray() {}

    public MockNDArray(
            NDManager manager,
            Device device,
            Shape shape,
            DataType dataType,
            SparseFormat sparseFormat) {
        this.manager = manager;
        this.device = device;
        this.shape = shape;
        this.dataType = dataType;
        this.sparseFormat = sparseFormat;
        uid = UUID.randomUUID().toString();
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
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return sparseFormat;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return sparseFormat != SparseFormat.DENSE;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {}

    @Override
    public void attachGradient(SparseFormat sparseFormat) {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        data.rewind();
        return data;
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        int size = data.remaining();
        // int8, uint8, boolean use ByteBuffer, so need to explicitly input DataType
        DataType inputType = DataType.fromBuffer(data);

        int numOfBytes = inputType.getNumOfBytes();
        this.data = ByteBuffer.allocate(size * numOfBytes);

        switch (inputType) {
            case FLOAT32:
                this.data.asFloatBuffer().put((FloatBuffer) data);
                break;
            case FLOAT64:
                this.data.asDoubleBuffer().put((DoubleBuffer) data);
                break;
            case INT8:
                this.data.put((ByteBuffer) data);
                break;
            case INT32:
                this.data.asIntBuffer().put((IntBuffer) data);
                break;
            case INT64:
                this.data.asLongBuffer().put((LongBuffer) data);
                break;
            case UINT8:
            case FLOAT16:
            default:
                throw new AssertionError("Show never happen");
        }
    }

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
        NDIndexFixed ie = (NDIndexFixed) index.get(0);
        int idx = (int) ie.getIndex();

        Shape subShape = shape.slice(1);
        int size = (int) subShape.size() * dataType.getNumOfBytes();
        int start = idx * size;
        data.position(start);
        Buffer buf = data.slice().limit(size);
        MockNDArray array = new MockNDArray(manager, device, subShape, dataType, sparseFormat);
        array.set(buf);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {}

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        return null;
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
    public NDArray eq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        return null;
    }
    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        return null;
    }
    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        return null;
    }
    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray others) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        Number[] values = toArray();
        switch (dataType) {
            case FLOAT64:
                {
                    double[] newBuf = new double[values.length];
                    for (int i = 0; i < values.length; ++i) {
                        newBuf[i] = Math.abs(values[i].doubleValue());
                    }
                    return manager.create(newBuf, shape);
                }
            case FLOAT32:
                {
                    float[] newBuf = new float[values.length];
                    for (int i = 0; i < values.length; ++i) {
                        newBuf[i] = Math.abs(values[i].floatValue());
                    }
                    return manager.create(newBuf, shape);
                }
            case INT64:
                {
                    long[] newBuf = new long[values.length];
                    for (int i = 0; i < values.length; ++i) {
                        newBuf[i] = Math.abs(values[i].longValue());
                    }
                    return manager.create(newBuf, shape);
                }
            case INT32:
                {
                    int[] newBuf = new int[values.length];
                    for (int i = 0; i < values.length; ++i) {
                        newBuf[i] = Math.abs(values[i].intValue());
                    }
                    return manager.create(newBuf, shape);
                }
            default:
                return null;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long sections, int axis) {
        NDList list = new NDList();
        for (int i = 0; i < sections; i++) {
            list.add(new MockNDArray(manager, device, getShape().slice(1), dataType, sparseFormat));
        }
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshapeLike(NDArray array) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        return new MockNDArray();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, float temperature) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int[] axes, float temperature) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(NDIndex index) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray swapAxes(int axis1, int axis2) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean shapeEquals(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return new MockNDArrayEx(this);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {}
}
