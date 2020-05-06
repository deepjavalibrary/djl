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
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.index.NDIndexBooleans;
import ai.djl.ndarray.index.NDIndexElement;
import ai.djl.ndarray.index.NDIndexFullSlice;
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
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Max;
import org.tensorflow.op.core.Min;
import org.tensorflow.op.core.Prod;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.core.Sum;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.TopK;
import org.tensorflow.tools.buffer.ByteDataBuffer;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

public class TfNDArray implements NDArray {

    private static final int MAX_SIZE = 100;
    private static final int MAX_DEPTH = 10;
    private static final int MAX_ROWS = 10;
    private static final int MAX_COLUMNS = 20;
    private static final int MAX_OUTPUTS_PER_OP = 8;

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
        this.shape = new Shape(tensor.shape().asArray());
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    TfNDArray(NDManager manager, Operand<?> out) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        this.tensor = out.asOutput().tensor();
        this.shape = new Shape(tensor.shape().asArray());
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    public TfNDArray(NDManager manager, Shape shape, FloatBuffer data) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        tensor = Tensor.of(TFloat32.DTYPE, toTfShape(shape), toDataBuffer(data));
        this.shape = shape;
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    TfNDArray(NDManager manager, Shape shape, ByteBuffer data) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        this.shape = shape;
        this.tf = this.manager.getTf();
        tensor = Tensor.of(TUint8.DTYPE, toTfShape(shape), DataBuffers.of(data));
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
            shape = new Shape(tensor.shape().asArray());
        }
        return shape;
    }

    public org.tensorflow.DataType<? extends TType> getTfDataType() {
        return tensor.dataType();
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
        Operand<?> output = tf.dtypes.cast(asOperand(), TfDataType.toTf(dataType));
        if (copy) {
            output = tf.deepCopy(output);
        }
        return new TfNDArray(manager, output);
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
    public double[] toDoubleArray() {
        double[] result = new double[(int) getShape().size()];
        tensor.rawData().asDoubles().read(result);
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        float[] result = new float[(int) getShape().size()];
        tensor.rawData().asFloats().read(result);
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public int[] toIntArray() {
        int[] result = new int[(int) getShape().size()];
        tensor.rawData().asInts().read(result);
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public long[] toLongArray() {
        long[] result = new long[(int) getShape().size()];
        tensor.rawData().asLongs().read(result);
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public boolean[] toBooleanArray() {
        boolean[] result = new boolean[(int) getShape().size()];
        tensor.rawData().asBooleans().read(result);
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
        tensor.rawData().read(buf);
        return ByteBuffer.wrap(buf);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, NDArray value) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Number value) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void setScalar(NDIndex index, Number value) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        if (index.getRank() == 0 && getShape().isScalar()) {
            return this;
        }
        // use booleanMask for NDIndexBooleans case
        List<NDIndexElement> indices = index.getIndices();
        if (!indices.isEmpty() && indices.get(0) instanceof NDIndexBooleans) {
            if (indices.size() != 1) {
                throw new IllegalArgumentException(
                        "get() currently didn't support more that one boolean NDArray");
            }
            return booleanMask(((NDIndexBooleans) indices.get(0)).getIndex());
        }

        NDIndexFullSlice fullSlice = index.getAsFullSlice(getShape()).orElse(null);
        if (fullSlice != null) {
            long[] begin = fullSlice.getMin();
            long[] end = fullSlice.getMax();

            long[] size = new long[begin.length];
            Arrays.setAll(size, i -> end[i] - begin[i]);
            Operand<?> sliced = tf.slice(asOperand(), tf.constant(begin), tf.constant(size));
            if (!fullSlice.getToSqueeze().isEmpty()) {
                sliced =
                        tf.squeeze(
                                sliced,
                                Squeeze.axis(
                                        fullSlice
                                                .getToSqueeze()
                                                .stream()
                                                .map(Integer::longValue)
                                                .collect(Collectors.toList())));
            }
            return new TfNDArray(manager, sliced);
        }
        throw new UnsupportedOperationException(
                "get() currently supports all, fixed, and slices indices");
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
        ((TfNDArray) ndArray).tensor = tf.deepCopy(asOperand()).asOutput().tensor();
        ((TfNDArray) ndArray).operand = null;
        ((TfNDArray) ndArray).shape = new Shape(tensor.shape().asArray());
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
        return new TfNDArray(
                manager,
                tf.gather(
                        asOperand(),
                        tf.squeeze(
                                tf.where(((TfNDArray) index).asOperand()),
                                Squeeze.axis(Collections.singletonList((long) 1))),
                        tf.constant(axis)));
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
        return new TfNDArray(manager, tf.zerosLike(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return new TfNDArray(manager, tf.onesLike(asOperand()));
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
    public NDArray eq(Number other) {
        return eq(manager.create(other).toType(getDataType(), false));
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
        return neq(manager.create(other).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return new TfNDArray(
                manager, tf.math.notEqual(asOperand(), ((TfNDArray) other).asOperand()).asOutput());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        return gt(manager.create(other).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return new TfNDArray(
                manager, tf.math.greater(asOperand(), ((TfNDArray) other).asOperand()).asOutput());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        return gte(manager.create(other).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return new TfNDArray(
                manager,
                tf.math.greaterEqual(asOperand(), ((TfNDArray) other).asOperand()).asOutput());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        return lt(manager.create(other).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return new TfNDArray(
                manager, tf.math.less(asOperand(), ((TfNDArray) other).asOperand()).asOutput());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        return lte(manager.create(other).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return new TfNDArray(
                manager,
                tf.math.lessEqual(asOperand(), ((TfNDArray) other).asOperand()).asOutput());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray all() {
        // TF takes bool for reduce and INT64 for indices
        return new TfNDArray(
                manager,
                tf.reduceAll(
                        tf.dtypes.cast(asOperand(), TBool.DTYPE),
                        tf.range(tf.constant(0L), tf.constant((long) getRank()), tf.constant(1L))));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray any() {
        // TF takes bool for reduce and INT64 for indices
        return new TfNDArray(
                manager,
                tf.reduceAny(
                        tf.dtypes.cast(asOperand(), TBool.DTYPE),
                        tf.range(tf.constant(0L), tf.constant((long) getRank()), tf.constant(1L))));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return add(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return new TfNDArray(manager, tf.math.add(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return sub(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return new TfNDArray(manager, tf.math.sub(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return mul(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return new TfNDArray(manager, tf.math.mul(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return div(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return new TfNDArray(manager, tf.math.div(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return mod(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return new TfNDArray(manager, tf.math.mod(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        return pow(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        return new TfNDArray(manager, tf.math.pow(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        return maximum(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        return new TfNDArray(
                manager, tf.math.maximum(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        return minimum(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        return new TfNDArray(
                manager, tf.math.minimum(asOperand(), ((TfNDArray) other).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return addi(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return inPlaceHelper(add(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return subi(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return inPlaceHelper(sub(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return muli(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        return inPlaceHelper(mul(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return divi(manager.create(n).toType(getDataType(), false));
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

        // inplace update destination tensor and operand
        ((TfNDArray) destination)
                .setTensor(
                        tf.inplaceUpdate(
                                        ((TfNDArray) destination).asOperand(),
                                        indices,
                                        ((TfNDArray) source).asOperand())
                                .asOutput()
                                .tensor());
        ((TfNDArray) destination).clearOperand();

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
        return modi(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return inPlaceHelper(mod(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        return powi(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        return inPlaceHelper(pow(other), this);
    }

    NDArray rpowi(NDArray other) {
        return inPlaceHelper(other.pow(this), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return new TfNDArray(manager, tf.math.neg(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return inPlaceHelper(neg(), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return new TfNDArray(manager, tf.math.abs(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return new TfNDArray(manager, tf.math.square(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        return new TfNDArray(manager, tf.math.sqrt(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return new TfNDArray(manager, tf.math.pow(asOperand(), toConstant(1f / 3, getDataType())));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return new TfNDArray(manager, tf.math.floor(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return new TfNDArray(manager, tf.math.ceil(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return new TfNDArray(manager, tf.math.round(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return new TfNDArray(manager, tf.math.exp(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return new TfNDArray(manager, tf.math.log(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return new TfNDArray(
                manager,
                tf.math.div(tf.math.log(asOperand()), tf.math.log(toConstant(10, getDataType()))));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return new TfNDArray(
                manager,
                tf.math.div(tf.math.log(asOperand()), tf.math.log(toConstant(2, getDataType()))));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return new TfNDArray(manager, tf.math.sin(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return new TfNDArray(manager, tf.math.cos(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return new TfNDArray(manager, tf.math.tan(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return new TfNDArray(manager, tf.math.asin(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return new TfNDArray(manager, tf.math.acos(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return new TfNDArray(manager, tf.math.atan(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return new TfNDArray(manager, tf.math.sinh(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return new TfNDArray(manager, tf.math.cosh(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return new TfNDArray(manager, tf.math.tanh(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return new TfNDArray(manager, tf.math.asinh(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return new TfNDArray(manager, tf.math.acosh(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return new TfNDArray(manager, tf.math.atanh(asOperand()));
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
        return new TfNDArray(
                manager,
                tf.max(asOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return new TfNDArray(
                manager, tf.max(asOperand(), tf.constant(axes), Max.keepDims(keepDims)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        return new TfNDArray(
                manager,
                tf.min(asOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return new TfNDArray(
                manager, tf.min(asOperand(), tf.constant(axes), Min.keepDims(keepDims)));
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public NDArray sum() {
        // sum on all axis
        Operand op;
        // tf can't sum boolean values
        if (getDataType() == DataType.BOOLEAN) {
            op = tf.dtypes.cast(asOperand(), TInt64.DTYPE);
        } else {
            op = asOperand();
        }
        return new TfNDArray(
                manager,
                tf.sum(
                        op,
                        tf.range(tf.constant(0L), tf.constant((long) getRank()), tf.constant(1L))));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return new TfNDArray(
                manager,
                tf.sum(
                        asOperand(),
                        ((TfNDArray) manager.create(axes)).asOperand(),
                        Sum.keepDims(keepDims)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        return new TfNDArray(
                manager,
                tf.prod(
                        asOperand(),
                        tf.range(tf.constant(0L), tf.constant((long) getRank()), tf.constant(1L))));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return new TfNDArray(
                manager,
                tf.prod(asOperand(), tf.constant(axes), Prod.keepDims(keepDims)).asOutput());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return new TfNDArray(
                manager,
                tf.math.mean(
                        asOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return new TfNDArray(
                manager,
                tf.math.mean(asOperand(), tf.constant(axes), Mean.keepDims(keepDims)).asOutput());
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
                for (int i = 0; i < MAX_OUTPUTS_PER_OP - 1; i++) {
                    partialIndices[i] = indices[start + i];
                }
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
            for (int i = 0; i < partialIndices.length; i++) {
                partialIndices[i] = indices[start + i];
            }
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
                        asOperand(),
                        tf.constant(sizes.stream().mapToInt(Long::intValue).toArray()),
                        tf.constant(axis),
                        (long) sizes.size())
                .forEach(output -> result.add(new TfNDArray(manager, output)));
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
        return new TfNDArray(manager, tf.reshape(asOperand(), tf.constant(shape.getShape())));
    }

    /** {@inheritDoc} */
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
        if (isScalar()) {
            axes = new int[0];
        }
        return new TfNDArray(
                manager,
                tf.squeeze(
                        asOperand(),
                        Squeeze.axis(
                                Arrays.stream(axes)
                                        .mapToLong(i -> (long) i)
                                        .boxed()
                                        .collect(Collectors.toList()))));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray n) {
        return new TfNDArray(
                manager,
                tf.math.logicalAnd(
                        tf.dtypes.cast(asOperand(), TBool.DTYPE),
                        tf.dtypes.cast(((TfNDArray) n).asOperand(), TBool.DTYPE)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray n) {
        return new TfNDArray(
                manager,
                tf.math.logicalOr(
                        tf.dtypes.cast(asOperand(), TBool.DTYPE),
                        tf.dtypes.cast(((TfNDArray) n).asOperand(), TBool.DTYPE)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray n) {
        return logicalOr(n).logicalAnd(logicalAnd(n).logicalNot());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return new TfNDArray(manager, tf.math.logicalNot(tf.dtypes.cast(asOperand(), TBool.DTYPE)));
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
            input = asOperand();
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
            input = tf.linalg.transpose(asOperand(), ((TfNDArray) transposition).asOperand());
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
            result = tf.linalg.transpose(result, ((TfNDArray) transposition).asOperand());
            transposition.close();
        }
        // re-apply neg after sort if ascending
        if (ascending && !returnIndices) {
            result = tf.math.neg(result);
        }
        return new TfNDArray(manager, result);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, float temperature) {
        if (temperature != 1.0) {
            throw new UnsupportedOperationException(
                    "TensorFlow softmax didn't suuport temperature");
        }
        return new TfNDArray(manager, softmaxHelper(axes, false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int[] axes, float temperature) {
        if (temperature != 1.0) {
            throw new UnsupportedOperationException(
                    "TensorFlow softmax didn't suuport temperature");
        }
        return new TfNDArray(manager, softmaxHelper(axes, true));
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private Operand softmaxHelper(int[] axes, boolean logSoftmax) {
        long dim = getShape().dimension();
        // if axis is -1 or last dim, directly apply softmax
        if (axes.length > 1) {
            throw new UnsupportedOperationException(
                    "TensorFlow softmax does not support multiple axes");
        }
        // return itself if zero dim
        if (dim == 0) {
            return asOperand();
        }
        if (axes[0] == -1 || axes[0] == dim - 1) {
            return logSoftmax ? tf.nn.logSoftmax(asOperand()) : tf.nn.softmax(asOperand());
        }
        if (axes[0] < -dim || axes[0] >= dim) {
            throw new IllegalArgumentException(
                    "Invalid axes value: "
                            + axes[0]
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
        concatList.add(tf.range(tf.constant(0L), tf.constant(axes[0] % dim), tf.constant(1L)));
        concatList.add(tf.expandDims(tf.constant(dim - 1), tf.constant(0)));
        concatList.add(
                tf.range(tf.constant((long) axes[0] + 1), tf.constant(dim - 1), tf.constant(1L)));
        concatList.add(tf.expandDims(tf.constant((long) axes[0]), tf.constant(0)));

        Operand transposed =
                tf.linalg.transpose(asOperand(), tf.concat(concatList, tf.constant(0)));
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
        return new TfNDArray(manager, tf.math.cumsum(asOperand(), tf.constant(axis)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return cumSum(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return new TfNDArray(manager, tf.dtypes.cast(tf.math.isInf(asOperand()), TBool.DTYPE));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return new TfNDArray(manager, tf.dtypes.cast(tf.math.isNan(asOperand()), TBool.DTYPE));
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
        return new TfNDArray(manager, tf.tile(asOperand(), tf.constant(repeats)));
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
            return new TfNDArray(
                    manager, tf.train.batchMatMul(asOperand(), ((TfNDArray) other).asOperand()));
        }
        Operand lhs = asOperand();
        Operand rhs = ((TfNDArray) other).asOperand();
        boolean broadcast = false;
        if (getShape().dimension() == 1) {
            lhs = tf.broadcastTo(asOperand(), tf.constant(new long[] {1L, getShape().get(0)}));
            broadcast = true;
        }

        if (other.getShape().dimension() == 1) {
            rhs =
                    tf.broadcastTo(
                            ((TfNDArray) other).asOperand(),
                            tf.constant(new long[] {1L, getShape().get(0)}));
            broadcast = true;
        }
        if (broadcast) {
            return new TfNDArray(manager, tf.linalg.matMul(lhs, rhs)).squeeze();
        } else {
            return new TfNDArray(manager, tf.linalg.matMul(lhs, rhs));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        return new TfNDArray(
                manager,
                tf.clipByValue(
                        asOperand(),
                        toConstant(min, getDataType()),
                        toConstant(max, getDataType())));
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
        return new TfNDArray(manager, tf.linalg.transpose(asOperand(), tf.constant(dimensions)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        return new TfNDArray(manager, tf.broadcastTo(asOperand(), tf.constant(shape.getShape())));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty NDArray");
        }
        return flatten().argMax(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        if (isScalar()) {
            return manager.create(0L);
        }
        return new TfNDArray(manager, tf.math.argMax(asOperand(), tf.constant(axis)));
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
        return new TfNDArray(manager, tf.math.argMin(asOperand(), tf.constant(axis)));
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
        if (tensor == null) {
            return "This array is already closed";
        }

        return toDebugString(MAX_SIZE, MAX_DEPTH, MAX_ROWS, MAX_COLUMNS);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (tensor != null) {
            tensor.close();
        }
        tensor = null;
        tf = null;
        operand = null;
        tfNDArrayEx = null;
    }

    @SuppressWarnings("unchecked")
    <T extends TType> Operand<T> asOperand() {
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

    void setTensor(Tensor<?> tensor) {
        this.tensor = tensor;
    }

    void clearOperand() {
        this.operand = null;
    }

    int getRank() {
        return tf.rank(asOperand()).asOutput().tensor().rawData().asInts().getInt(0);
    }

    private <T extends TType> Constant<T> toConstant(Number n, DataType jType) {
        return getConstant(n, jType, tf);
    }

    public static org.tensorflow.tools.Shape toTfShape(Shape shape) {
        return org.tensorflow.tools.Shape.of(shape.getShape());
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
