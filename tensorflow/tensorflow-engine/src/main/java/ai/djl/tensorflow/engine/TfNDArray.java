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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Max;
import org.tensorflow.op.core.Min;
import org.tensorflow.op.core.Prod;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.core.Sum;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.TopK;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

public class TfNDArray implements NDArray {

    private static final int MAX_SIZE = 100;
    private static final int MAX_DEPTH = 10;
    private static final int MAX_ROWS = 10;
    private static final int MAX_COLUMNS = 20;
    private static final int MAX_OUTPUTS_PER_OP = 1000;

    private String uid;
    private Shape shape;
    private TfNDManager manager;
    private Ops tf;
    private Operand<?> operand;
    private String name;
    private TfNDArrayEx tfNDArrayEx;
    private DataType dataType;

    TfNDArray(NDManager manager, Tensor<?> tensor) {
        this.manager = (TfNDManager) manager;
        this.tf = this.manager.getTf();
        uid = UUID.randomUUID().toString();
        manager.attach(uid, this);
        this.operand =
                this.manager
                        .getEagerSession()
                        .opBuilder("Const", "Const_" + TfNDManager.nextNameAssignment())
                        .setAttr("dtype", tensor.dataType())
                        .setAttr("value", tensor)
                        .setDevice(getTfDevice())
                        .build()
                        .output(0);
        // cache shape and data type information so we can close tensor later
        this.shape = new Shape(tensor.shape().asArray());
        this.dataType = TfDataType.fromTf(tensor.dataType());
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
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return manager.getDevice();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        Operand<?> output = tf.dtypes.cast(getOperand(), TfDataType.toTf(dataType));
        if (copy) {
            output = tf.deepCopy(output);
        }
        try (Tensor<?> tensor = output.asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {}

    /** {@inheritDoc} */
    @Override
    public void attachGradient(SparseFormat sparseFormat) {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasGradient() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public double[] toDoubleArray() {
        double[] result = new double[(int) getShape().size()];
        try (Tensor<?> tensor = operand.asTensor()) {
            tensor.rawData().asDoubles().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        float[] result = new float[(int) getShape().size()];
        try (Tensor<?> tensor = operand.asTensor()) {
            tensor.rawData().asFloats().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public int[] toIntArray() {
        int[] result = new int[(int) getShape().size()];
        try (Tensor<?> tensor = operand.asTensor()) {
            tensor.rawData().asInts().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public long[] toLongArray() {
        long[] result = new long[(int) getShape().size()];
        try (Tensor<?> tensor = operand.asTensor()) {
            tensor.rawData().asLongs().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public boolean[] toBooleanArray() {
        boolean[] result = new boolean[(int) getShape().size()];
        try (Tensor<?> tensor = operand.asTensor()) {
            tensor.rawData().asBooleans().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        Shape sh = getShape();
        DataType dType = getDataType();
        long product = sh.size();
        long len = dType.getNumOfBytes() * product;
        byte[] buf = new byte[Math.toIntExact(len)];
        try (Tensor<?> tensor = operand.asTensor()) {
            tensor.rawData().read(buf);
        }
        return ByteBuffer.wrap(buf);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public NDManager attach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (TfNDManager) manager;
        manager.attach(uid, this);
        return original;
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detach(getUid());
        manager = TfNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray ndArray) {
        if (!(ndArray instanceof TfNDArray)) {
            throw new IllegalArgumentException("Only TfNDArray is supported.");
        }
        Shape inShape = getShape();
        Shape destShape = ndArray.getShape();
        if (!Arrays.equals(inShape.getShape(), destShape.getShape())) {
            throw new IllegalArgumentException(
                    "shape are diff. Required: " + destShape + ", Actual " + inShape);
        }
        ((TfNDArray) ndArray).operand = tf.deepCopy(getOperand()).asOutput();
        ((TfNDArray) ndArray).dataType = getDataType();
        ((TfNDArray) ndArray).shape =
                new Shape(getShape().stream().mapToLong(pair -> pair.getKey()).toArray());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        // handle scalar case manually to behave like numpy
        if (isScalar()) {
            if (!index.isScalar()) {
                throw new IllegalArgumentException("Input is scalar, index must also be scalar.");
            }
            if (index.toBooleanArray()[0]) {
                return expandDims(0);
            } else {
                return manager.create(new Shape());
            }
        }
        try (Tensor<?> tensor =
                tf.gather(
                                getOperand(),
                                tf.squeeze(
                                        tf.where(((TfNDArray) index).getOperand()),
                                        Squeeze.axis(Collections.singletonList((long) 1))),
                                tf.constant(axis))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        try (Tensor<?> tensor = tf.zerosLike(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        try (Tensor<?> tensor = tf.onesLike(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
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
        TfNDArray eq = (TfNDArray) eq(other);
        return eq.all().toBooleanArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return eq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.equal(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return neq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.notEqual(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return gt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.greater(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return gte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.greaterEqual(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return lt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.less(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return lte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.lessEqual(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray all() {
        // TF takes bool for reduce and INT64 for indices
        try (Tensor<?> tensor =
                tf.reduceAll(
                                tf.dtypes.cast(getOperand(), TBool.DTYPE),
                                tf.range(
                                        tf.constant(0L),
                                        tf.constant((long) getRank()),
                                        tf.constant(1L)))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray any() {
        // TF takes bool for reduce and INT64 for indices
        try (Tensor<?> tensor =
                tf.reduceAny(
                                tf.dtypes.cast(getOperand(), TBool.DTYPE),
                                tf.range(
                                        tf.constant(0L),
                                        tf.constant((long) getRank()),
                                        tf.constant(1L)))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return add(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.add(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return sub(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.sub(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return mul(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.mul(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return div(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.div(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return mod(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.mod(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return pow(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.pow(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return maximum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.maximum(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return minimum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        try (Tensor<?> tensor =
                tf.math.minimum(getOperand(), ((TfNDArray) other).getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return addi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return inPlaceHelper(add(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return subi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return inPlaceHelper(sub(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return muli(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        return inPlaceHelper(mul(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return divi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return inPlaceHelper(div(other), this);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    NDArray inPlaceHelper(NDArray source, NDArray destination) {
        if (getShape().isScalar()) {
            throw new UnsupportedOperationException(
                    "TensorFlow engine does not support inplace operations on scalars yet");
        }
        // select all indices for inplace update
        Operand indices =
                tf.range(
                        tf.constant(0),
                        tf.constant((int) getShape().getShape()[0]),
                        tf.constant(1));

        // inplace update destination operand
        ((TfNDArray) destination)
                .setOperand(
                        tf.inplaceUpdate(
                                        ((TfNDArray) destination).getOperand(),
                                        indices,
                                        ((TfNDArray) source).getOperand())
                                .asOutput());

        return destination;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return modi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return inPlaceHelper(mod(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return powi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        return inPlaceHelper(pow(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sign() {
        try (Tensor<?> tensor = tf.math.sign(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray signi() {
        return inPlaceHelper(sign(), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        try (Tensor<?> tensor = tf.math.neg(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return inPlaceHelper(neg(), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        try (Tensor<?> tensor = tf.math.abs(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        try (Tensor<?> tensor = tf.math.square(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        try (Tensor<?> tensor = tf.math.sqrt(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        try (Tensor<?> tensor =
                tf.math.pow(getOperand(), toConstant(1f / 3, getDataType())).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        try (Tensor<?> tensor = tf.math.floor(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        try (Tensor<?> tensor = tf.math.ceil(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        try (Tensor<?> tensor = tf.math.round(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        try (Tensor<?> tensor = tf.math.exp(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        try (Tensor<?> tensor = tf.math.log(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        try (Tensor<?> tensor =
                tf.math
                        .div(tf.math.log(getOperand()), tf.math.log(toConstant(10, getDataType())))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        try (Tensor<?> tensor =
                tf.math
                        .div(tf.math.log(getOperand()), tf.math.log(toConstant(2, getDataType())))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        try (Tensor<?> tensor = tf.math.sin(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        try (Tensor<?> tensor = tf.math.cos(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        try (Tensor<?> tensor = tf.math.tan(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        try (Tensor<?> tensor = tf.math.asin(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        try (Tensor<?> tensor = tf.math.acos(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        try (Tensor<?> tensor = tf.math.atan(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        try (Tensor<?> tensor = tf.math.sinh(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        try (Tensor<?> tensor = tf.math.cosh(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        try (Tensor<?> tensor = tf.math.tanh(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        try (Tensor<?> tensor = tf.math.asinh(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        try (Tensor<?> tensor = tf.math.acosh(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        try (Tensor<?> tensor = tf.math.atanh(getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return mul(180).div(Math.PI);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return mul(Math.PI).div(180);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        try (Tensor<?> tensor =
                tf.max(getOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).getOperand())
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        try (Tensor<?> tensor =
                tf.max(getOperand(), tf.constant(axes), Max.keepDims(keepDims)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        try (Tensor<?> tensor =
                tf.min(getOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).getOperand())
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        try (Tensor<?> tensor =
                tf.min(getOperand(), tf.constant(axes), Min.keepDims(keepDims)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public NDArray sum() {
        // sum on all axis
        Operand op;
        // tf can't sum boolean values
        if (getDataType() == DataType.BOOLEAN) {
            op = tf.dtypes.cast(getOperand(), TInt64.DTYPE);
        } else {
            op = getOperand();
        }
        try (Tensor<?> tensor =
                tf.sum(
                                op,
                                tf.range(
                                        tf.constant(0L),
                                        tf.constant((long) getRank()),
                                        tf.constant(1L)))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        try (Tensor<?> tensor =
                tf.sum(
                                getOperand(),
                                ((TfNDArray) manager.create(axes)).getOperand(),
                                Sum.keepDims(keepDims))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        try (Tensor<?> tensor =
                tf.prod(
                                getOperand(),
                                tf.range(
                                        tf.constant(0L),
                                        tf.constant((long) getRank()),
                                        tf.constant(1L)))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        try (Tensor<?> tensor =
                tf.prod(getOperand(), tf.constant(axes), Prod.keepDims(keepDims))
                        .asOutput()
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        try (TfNDArray array = ((TfNDArray) manager.arange(0, getRank(), 1));
                Tensor<?> tensor = tf.math.mean(getOperand(), array.getOperand()).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        try (Tensor<?> tensor =
                tf.math.mean(getOperand(), tf.constant(axes), Mean.keepDims(keepDims)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        if (indices.length > MAX_OUTPUTS_PER_OP) {
            // split MAX_OUTPUTS_PER_OP -1 slices multiple times
            NDList result = new NDList();
            long totalSize = getShape().get(axis);
            int start = 0;

            while (start < indices.length - MAX_OUTPUTS_PER_OP + 2) {
                long[] partialIndices = new long[MAX_OUTPUTS_PER_OP];
                System.arraycopy(indices, start, partialIndices, 0, MAX_OUTPUTS_PER_OP - 1);
                partialIndices[MAX_OUTPUTS_PER_OP - 1] = totalSize;
                NDList splitted = splitHelper(partialIndices, axis);
                // remove last chunk from result
                splitted.remove(splitted.get(splitted.size() - 1));
                // remove first chunk from result
                if (start > 0) {
                    splitted.remove(splitted.get(0));
                }
                result.addAll(splitted);
                start += MAX_OUTPUTS_PER_OP - 2;
            }

            long[] partialIndices = new long[indices.length - start];
            System.arraycopy(indices, start, partialIndices, 0, partialIndices.length);
            NDList splitted = splitHelper(partialIndices, axis);
            // remove the first chunk from result
            splitted.remove(splitted.get(0));

            result.addAll(splitted);

            return result;
        } else {
            return splitHelper(indices, axis);
        }
    }

    // workaround for split output must be less than MAX_OUTPUTS_PER_OP
    // TODO: remove helper once this issue is fixed:
    // https://github.com/tensorflow/java/issues/45
    private NDList splitHelper(long[] indices, int axis) {
        if (indices.length == 0) {
            return new NDList(this);
        }
        NDList result = new NDList();

        List<Long> sizes = new ArrayList<>();
        int lastIndex = indices.length - 1;
        long dimSize = getShape().get(axis);
        // does not add to size if indices starts at 0
        if (indices[0] > 0) {
            sizes.add(indices[0]);
        }
        for (int i = 1; i < indices.length; i++) {
            sizes.add(indices[i] - indices[i - 1]);
        }
        // add last size if last index smaller than max size of that axis
        if (indices[lastIndex] < dimSize) {
            sizes.add(dimSize - indices[lastIndex]);
        }
        long totalSize = sizes.stream().mapToLong(Long::longValue).sum();

        if (totalSize != getShape().get(axis)) {
            throw new IllegalArgumentException(
                    "split sizes :"
                            + totalSize
                            + " must sum to dimension on axis "
                            + axis
                            + ": "
                            + getShape().get(axis));
        }

        tf.splitV(
                        getOperand(),
                        tf.constant(sizes.stream().mapToInt(Long::intValue).toArray()),
                        tf.constant(axis),
                        (long) sizes.size())
                .forEach(
                        output -> {
                            try (Tensor<?> tensor = output.asTensor()) {
                                result.add(new TfNDArray(manager, tensor));
                            }
                        });
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return reshape(new Shape(-1));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        try (Tensor<?> tensor =
                tf.reshape(getOperand(), tf.constant(shape.getShape())).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        try (Tensor<?> tensor = tf.expandDims(getOperand(), tf.constant(axis)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        if (isScalar()) {
            axes = new int[0];
        }
        try (Tensor<?> tensor =
                tf.squeeze(
                                getOperand(),
                                Squeeze.axis(
                                        Arrays.stream(axes)
                                                .mapToLong(i -> (long) i)
                                                .boxed()
                                                .collect(Collectors.toList())))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray n) {
        try (Tensor<?> tensor =
                tf.math
                        .logicalAnd(
                                tf.dtypes.cast(getOperand(), TBool.DTYPE),
                                tf.dtypes.cast(((TfNDArray) n).getOperand(), TBool.DTYPE))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray n) {
        try (Tensor<?> tensor =
                tf.math
                        .logicalOr(
                                tf.dtypes.cast(getOperand(), TBool.DTYPE),
                                tf.dtypes.cast(((TfNDArray) n).getOperand(), TBool.DTYPE))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray n) {
        return logicalOr(n).logicalAnd(logicalAnd(n).logicalNot());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        try (Tensor<?> tensor =
                tf.math.logicalNot(tf.dtypes.cast(getOperand(), TBool.DTYPE)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return sortHelper(axis, ascending, true);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return sortHelper(axis, true, false);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return sortHelper(-1, true, false);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private NDArray sortHelper(int axis, boolean ascending, boolean returnIndices) {
        if (isScalar()) {
            return this;
        }
        // using topK to implement argSort
        int k;
        int rank = getRank();
        NDArray transposition;
        Operand input;
        Operand result;
        if (axis == -1 || axis + 1 == getShape().dimension()) {
            // last axis
            transposition = null;
            input = getOperand();
            long[] arrayShape = getShape().getShape();
            k = (int) arrayShape[arrayShape.length - 1];
        } else {
            k = (int) getShape().getShape()[axis];

            transposition =
                    NDArrays.concat(
                            new NDList(
                                    manager.arange(0, axis, 1, DataType.INT32, getDevice()),
                                    manager.create(new int[] {rank - 1}),
                                    manager.arange(
                                            axis + 1, rank - 1, 1, DataType.INT32, getDevice()),
                                    manager.create(new int[] {axis})));
            input = tf.linalg.transpose(getOperand(), ((TfNDArray) transposition).getOperand());
        }
        TopK topK;
        if (ascending) {
            topK = tf.nn.topK(tf.math.neg(input), tf.constant(k));
        } else {
            topK = tf.nn.topK(input, tf.constant(k));
        }
        if (returnIndices) {
            // always return long as indices type
            result = tf.dtypes.cast(topK.indices(), TInt64.DTYPE);
        } else {
            result = topK.values();
        }
        if (transposition != null) {
            result = tf.linalg.transpose(result, ((TfNDArray) transposition).getOperand());
            transposition.close();
        }
        // re-apply neg after sort if ascending
        if (ascending && !returnIndices) {
            result = tf.math.neg(result);
        }
        try (Tensor<?> tensor = result.asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int axis) {
        try (Tensor<?> tensor = softmaxHelper(axis, false).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int axis) {
        try (Tensor<?> tensor = softmaxHelper(axis, true).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private Operand softmaxHelper(int axis, boolean logSoftmax) {
        long dim = getShape().dimension();
        // if axis is -1 or last dim, directly apply softmax
        // return itself if zero dim
        if (dim == 0) {
            return getOperand();
        }
        if (axis == -1 || axis == dim - 1) {
            return logSoftmax ? tf.nn.logSoftmax(getOperand()) : tf.nn.softmax(getOperand());
        }
        if (axis < -dim || axis >= dim) {
            throw new IllegalArgumentException(
                    "Invalid axes value: "
                            + axis
                            + ", must be in range ["
                            + -dim
                            + ", "
                            + dim
                            + ") where "
                            + dim
                            + " is the number of dimensions in the input.");
        }

        // tf.softmax always apply on last dimension, transpose input to make axes[0] last dimension
        ArrayList<Operand<TInt64>> concatList = new ArrayList<>();
        concatList.add(tf.range(tf.constant(0L), tf.constant(axis % dim), tf.constant(1L)));
        concatList.add(tf.expandDims(tf.constant(dim - 1), tf.constant(0)));
        concatList.add(
                tf.range(tf.constant((long) axis + 1), tf.constant(dim - 1), tf.constant(1L)));
        concatList.add(tf.expandDims(tf.constant((long) axis), tf.constant(0)));

        Operand transposed =
                tf.linalg.transpose(getOperand(), tf.concat(concatList, tf.constant(0)));
        // apply softmax
        Operand output = logSoftmax ? tf.nn.logSoftmax(transposed) : tf.nn.softmax(transposed);
        // transfer back to original shape
        return tf.linalg.transpose(output, tf.concat(concatList, tf.constant(0)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        // just expand dim for scalar
        if (isScalar()) {
            return expandDims(0);
        }
        // return 0 shape if any of the dim is 0
        if (Arrays.stream(getShape().getShape()).anyMatch(dim -> dim == 0L)) {
            return manager.create(new Shape(0));
        }
        try (Tensor<?> tensor = tf.math.cumsum(getOperand(), tf.constant(axis)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return cumSum(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        try (Tensor<?> tensor =
                tf.dtypes.cast(tf.math.isInf(getOperand()), TBool.DTYPE).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        try (Tensor<?> tensor =
                tf.dtypes.cast(tf.math.isNan(getOperand()), TBool.DTYPE).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        long[] multiples = new long[getShape().dimension()];
        Arrays.fill(multiples, repeats);
        return tile(multiples);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        long[] multiples = new long[getShape().dimension()];
        Arrays.fill(multiples, 1);
        multiples[axis] = repeats;
        return tile(multiples);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        try (Tensor<?> tensor = tf.tile(getOperand(), tf.constant(repeats)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
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
    @SuppressWarnings({"rawtypes", "unchecked"})
    public NDArray matMul(NDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for matMul()");
        }
        if (getShape().dimension() > 2 || other.getShape().dimension() > 2) {
            try (Tensor<?> tensor =
                    tf.train
                            .batchMatMul(getOperand(), ((TfNDArray) other).getOperand())
                            .asTensor()) {
                return new TfNDArray(manager, tensor);
            }
        }
        Operand lhs = getOperand();
        Operand rhs = ((TfNDArray) other).getOperand();
        boolean broadcast = false;
        if (getShape().dimension() == 1) {
            lhs = tf.broadcastTo(getOperand(), tf.constant(new long[] {1L, getShape().get(0)}));
            broadcast = true;
        }

        if (other.getShape().dimension() == 1) {
            rhs =
                    tf.broadcastTo(
                            ((TfNDArray) other).getOperand(),
                            tf.constant(new long[] {1L, getShape().get(0)}));
            broadcast = true;
        }
        try (Tensor<?> tensor = tf.linalg.matMul(lhs, rhs).asTensor()) {
            TfNDArray result = new TfNDArray(manager, tensor);
            if (broadcast) {
                return result.squeeze();
            } else {
                return result;
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        try (Tensor<?> tensor =
                tf.clipByValue(
                                getOperand(),
                                toConstant(min, getDataType()),
                                toConstant(max, getDataType()))
                        .asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    @Override
    public NDArray flip(int... axes) {
        try (Tensor<?> tensor = tf.reverse(getOperand(), tf.constant(axes)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        int dim = getShape().dimension();
        int[] reversedShape = IntStream.range(0, dim).map(i -> dim - i - 1).toArray();
        return transpose(reversedShape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
        if (Arrays.stream(dimensions).anyMatch(d -> d < 0)) {
            throw new UnsupportedOperationException(
                    "Passing -1 for broadcasting the dimension is not currently supported");
        }
        if (!Arrays.equals(
                Arrays.stream(dimensions).sorted().toArray(),
                IntStream.range(0, getShape().dimension()).toArray())) {
            throw new IllegalArgumentException(
                    "You must include each of the dimensions from 0 until "
                            + getShape().dimension());
        }
        try (Tensor<?> tensor =
                tf.linalg.transpose(getOperand(), tf.constant(dimensions)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        try (Tensor<?> tensor =
                tf.broadcastTo(getOperand(), tf.constant(shape.getShape())).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty NDArray");
        }
        return flatten().argMax(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        if (isScalar()) {
            return manager.create(0L);
        }
        try (Tensor<?> tensor = tf.math.argMax(getOperand(), tf.constant(axis)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty NDArray");
        }
        return flatten().argMin(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        if (isScalar()) {
            return manager.create(0L);
        }
        try (Tensor<?> tensor = tf.math.argMin(getOperand(), tf.constant(axis)).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] dimension) {
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
    public NDArray nonzero() {
        throw new UnsupportedOperationException("Not implemented");
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
        if (operand == null) {
            return "This array is already closed";
        }

        return toDebugString(MAX_SIZE, MAX_DEPTH, MAX_ROWS, MAX_COLUMNS);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        tf = null;
        operand = null;
        tfNDArrayEx = null;
    }

    @SuppressWarnings("unchecked")
    <T extends TType> Operand<T> getOperand() {
        return (Operand<T>) operand;
    }

    private String getTfDevice() {
        if (getDevice().getDeviceType().equals(Device.Type.CPU)) {
            return "/device:CPU:0";
        } else if (getDevice().getDeviceType().equals(Device.Type.GPU)) {
            return "/device:GPU:" + getDevice().getDeviceId();
        } else {
            throw new EngineException(
                    "Unknown device type to TensorFlow Engine: " + getDevice().toString());
        }
    }

    public Tensor<?> getTensor() {
        return operand.asTensor();
    }

    void setOperand(Operand<?> operand) {
        this.operand = operand;
    }

    int getRank() {
        try (Tensor<?> tensor = tf.rank(getOperand()).asOutput().tensor()) {
            return tensor.rawData().asInts().getInt(0);
        }
    }

    private <T extends TType> Constant<T> toConstant(Number n, DataType jType) {
        return getConstant(n, jType, tf);
    }

    public static org.tensorflow.ndarray.Shape toTfShape(Shape shape) {
        return org.tensorflow.ndarray.Shape.of(shape.getShape());
    }

    public static ByteDataBuffer toDataBuffer(FloatBuffer buffer) {
        // FIXME: find a better way or improve TF java implemenetation
        ByteBuffer bb = ByteBuffer.allocate(buffer.remaining() * 4);
        bb.asFloatBuffer().put(buffer);
        return DataBuffers.of(bb);
    }

    @SuppressWarnings("unchecked")
    static <T extends TType> Constant<T> getConstant(Number n, DataType jType, Ops tf) {
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
