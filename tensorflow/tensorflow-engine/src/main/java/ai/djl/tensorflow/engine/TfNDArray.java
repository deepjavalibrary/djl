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
package ai.djl.tensorflow.engine;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.UUID;
import java.util.function.Predicate;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.types.UInt8;

public class TfNDArray implements NDArray {

    private static final int MAX_SIZE = 100;
    private static final int MAX_DEPTH = 10;
    private static final int MAX_ROWS = 10;
    private static final int MAX_COLUMNS = 20;

    private String uid = UUID.randomUUID().toString();
    private Tensor<?> tensor;
    private Shape shape;
    private TfNDManager manager;
    private Ops tf;
    private Operand<?> operand;
    private String name;
    private TfNDArrayEx tfNDArrayEx;

    TfNDArray(NDManager manager, Tensor<?> tensor) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        this.tensor = tensor;
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    TfNDArray(NDManager manager, Operand<?> out) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        this.tensor = out.asOutput().tensor();
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    public TfNDArray(NDManager manager, Shape shape, FloatBuffer data) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        tensor = Tensor.create(shape.getShape(), data);
        this.shape = shape;
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    TfNDArray(NDManager manager, Shape shape, ByteBuffer data) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        tensor = Tensor.create(UInt8.class, shape.getShape(), data);
        this.shape = shape;
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
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
    public final String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return TfDataType.fromTf(getTfDataType());
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return manager.getDevice();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            // runToTensor();
            shape = new Shape(tensor.shape());
        }
        return shape;
    }

    public org.tensorflow.DataType getTfDataType() {
        return tensor.dataType();
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        return new TfNDArray(
                manager, tf.dtypes.cast(asOperand(), TfDataType.toPrimitiveClass(dataType)));
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        Shape sh = getShape();
        DataType dType = getDataType();
        long product = sh.size();
        long len = dType.getNumOfBytes() * product;
        ByteBuffer bb = manager.allocateDirect(Math.toIntExact(len));
        tensor.writeTo(bb);
        // reset buffer position for converting to other buffer type
        bb.rewind();
        return bb;
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
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
        return null;
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
    public NDArray zerosLike() {
        return new TfNDArray(manager, tf.zerosLike(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return new TfNDArray(manager, tf.onesLike(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        if (number == null) {
            return false;
        }
        try (NDArray result = eq(number)) {
            return result.all().getBoolean();
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        if (other == null || (!shapeEquals(other))) {
            return false;
        }
        if (getDataType() != other.getDataType()) {
            return false;
        }
        return tf.reduceAll(
                        ((TfNDArray) eq(other)).asOperand(),
                        ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand())
                .asOutput()
                .tensor()
                .booleanValue();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return new TfNDArray(
                manager, tf.math.equal(asOperand(), ((TfNDArray) other).asOperand()).asOutput());
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
    public NDArray toSparse(SparseFormat fmt) {
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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return new TfNDArray(manager, tf.math.square(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return new TfNDArray(manager, tf.math.pow(asOperand(), toConstant(1f / 3, getDataType())));
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
    public NDList split(long[] indices, int axis) {
        return null;
    }

    @Override
    public NDList split(long sections, int axis) {
        NDList result = new NDList();
        tf.split(tf.constant(axis), operand, sections)
                .forEach(
                        output -> {
                            result.add(new TfNDArray(manager, output));
                        });
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return new TfNDArray(manager, tf.reshape(asOperand(), tf.constant(shape.getShape())));
    }

    @Override
    public NDArray reshapeLike(NDArray array) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        return new TfNDArray(manager, tf.expandDims(asOperand(), tf.constant(axis)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray n) {
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
    public NDArray sort(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return null;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override
    public NDArray softmax(int[] axes, float temperature) {
        return new TfNDArray(manager, tf.nn.softmax(asOperand()));
    }

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
    public NDArray argMax() {
        return argMax(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        return new TfNDArray(manager, tf.math.argMax(asOperand(), tf.constant(axis)));
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
    public NDArray median(int[] axes) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
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
        return tfNDArrayEx;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof TfNDArray) {
            return contentEquals((TfNDArray) obj);
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
    public String toString() {
        if (tensor == null) {
            return "This array is already closed";
        }
        return toDebugString(this, MAX_SIZE, MAX_DEPTH, MAX_ROWS, MAX_COLUMNS);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (tensor != null) {
            tensor.close();
        }
        tensor = null;
        tf = null;
    }

    @SuppressWarnings("unchecked")
    <T> Operand<T> asOperand() {
        if (operand == null) {
            Operation op =
                    manager.getEagerSession()
                            .opBuilder("Const", "Const_" + TfNDManager.nextNameAssignment())
                            .setAttr("dtype", tensor.dataType())
                            .setAttr("value", tensor)
                            .build();
            operand = op.output(0);
        }
        return (Operand<T>) operand;
    }

    public Tensor<?> getTensor() {
        return tensor;
    }

    int getRank() {
        return tf.rank(asOperand()).asOutput().tensor().intValue();
    }

    private <T> Constant<T> toConstant(Number n, DataType jType) {
        return getConstant(n, jType, tf);
    }

    @SuppressWarnings("unchecked")
    static <T> Constant<T> getConstant(Number n, DataType jType, Ops tf) {
        switch (jType) {
            case INT8:
                return (Constant<T>) tf.constant(n.byteValue());
            case INT32:
                return (Constant<T>) tf.constant(n.intValue());
            case INT64:
                return (Constant<T>) tf.constant(n.longValue());
            case FLOAT16:
                return (Constant<T>) tf.constant(n.shortValue());
            case FLOAT32:
                return (Constant<T>) tf.constant(n.floatValue());
            case FLOAT64:
                return (Constant<T>) tf.constant(n.doubleValue());
            default:
                throw new EngineException("unsupported type");
        }
    }
}
