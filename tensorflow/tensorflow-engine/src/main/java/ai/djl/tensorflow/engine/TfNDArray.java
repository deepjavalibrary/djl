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
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.TopK;
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
        this.shape = new Shape(tensor.shape());
        this.tf = this.manager.getTf();
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    TfNDArray(NDManager manager, Operand<?> out) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(getUid(), this);
        this.tensor = out.asOutput().tensor();
        this.shape = new Shape(tensor.shape());
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
        Operand<?> output = tf.dtypes.cast(asOperand(), TfDataType.toPrimitiveClass(dataType));
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
    public NDArray getGradient() {
        throw new UnsupportedOperationException("Not implemented");
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
        ((TfNDArray) ndArray).shape = new Shape(tensor.shape());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
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

    @Override
    public NDArray all() {
        return new TfNDArray(
                manager,
                tf.reduceAll(
                        asOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand()));
    }

    @Override
    public NDArray any() {
        return new TfNDArray(
                manager,
                tf.reduceAny(
                        asOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand()));
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
        if (getShape().isScalar()) {
            throw new UnsupportedOperationException(
                    "TensorFlow engine does not support inplace operations on scalars yet");
        }
        // from TensorFlow inplaceAdd documentation:
        // Adds v into specified rows of x., computes y = x; y[i, :] += v; return y.
        // i: A vector. Indices into the left-most dimension of x.
        // To apply full inplaceAdd here in DJL, specify i to be all the rows.
        tensor =
                tf.inplaceAdd(
                                asOperand(),
                                ((TfNDArray)
                                                manager.arange(
                                                        0,
                                                        getShape().getShape()[0],
                                                        1,
                                                        DataType.INT32,
                                                        getDevice()))
                                        .asOperand(),
                                ((TfNDArray) other).asOperand())
                        .asOutput()
                        .tensor();
        operand = null;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return subi(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        if (getShape().isScalar()) {
            throw new UnsupportedOperationException(
                    "TensorFlow engine does not support inplace operations on scalars yet");
        }
        tensor =
                tf.inplaceSub(
                                asOperand(),
                                ((TfNDArray)
                                                manager.arange(
                                                        0,
                                                        getShape().getShape()[0],
                                                        1,
                                                        DataType.INT32,
                                                        getDevice()))
                                        .asOperand(),
                                ((TfNDArray) other).asOperand())
                        .asOutput()
                        .tensor();
        operand = null;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return muli(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        try (NDArray result = mul(other)) {
            if (getShape().isScalar()) {
                throw new UnsupportedOperationException(
                        "TensorFlow engine does not support inplace operations on scalars yet");
            }
            tensor =
                    tf.inplaceUpdate(
                                    asOperand(),
                                    ((TfNDArray)
                                                    manager.arange(
                                                            0,
                                                            getShape().getShape()[0],
                                                            1,
                                                            DataType.INT32,
                                                            getDevice()))
                                            .asOperand(),
                                    ((TfNDArray) result).asOperand())
                            .asOutput()
                            .tensor();
            operand = null;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return divi(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        try (NDArray result = div(other)) {
            if (getShape().isScalar()) {
                throw new UnsupportedOperationException(
                        "TensorFlow engine does not support inplace operations on scalars yet");
            }
            tensor =
                    tf.inplaceUpdate(
                                    asOperand(),
                                    ((TfNDArray)
                                                    manager.arange(
                                                            0,
                                                            getShape().getShape()[0],
                                                            1,
                                                            DataType.INT32,
                                                            getDevice()))
                                            .asOperand(),
                                    ((TfNDArray) result).asOperand())
                            .asOutput()
                            .tensor();
            operand = null;
        }
        return this;
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
        try (NDArray result = mod(other)) {
            if (getShape().isScalar()) {
                throw new UnsupportedOperationException(
                        "TensorFlow engine does not support inplace operations on scalars yet");
            }
            tensor =
                    tf.inplaceUpdate(
                                    asOperand(),
                                    ((TfNDArray)
                                                    manager.arange(
                                                            0,
                                                            getShape().getShape()[0],
                                                            1,
                                                            DataType.INT32,
                                                            getDevice()))
                                            .asOperand(),
                                    ((TfNDArray) result).asOperand())
                            .asOutput()
                            .tensor();
            operand = null;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        return powi(manager.create(n).toType(getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        try (NDArray result = pow(other)) {
            if (getShape().isScalar()) {
                throw new UnsupportedOperationException(
                        "TensorFlow engine does not support inplace operations on scalars yet");
            }
            tensor =
                    tf.inplaceUpdate(
                                    asOperand(),
                                    ((TfNDArray)
                                                    manager.arange(
                                                            0,
                                                            getShape().getShape()[0],
                                                            1,
                                                            DataType.INT32,
                                                            getDevice()))
                                            .asOperand(),
                                    ((TfNDArray) result).asOperand())
                            .asOutput()
                            .tensor();
            operand = null;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return new TfNDArray(manager, tf.math.neg(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        try (NDArray result = neg()) {
            if (getShape().isScalar()) {
                throw new UnsupportedOperationException(
                        "TensorFlow engine does not support inplace operations on scalars yet");
            }
            tensor =
                    tf.inplaceUpdate(
                                    asOperand(),
                                    ((TfNDArray)
                                                    manager.arange(
                                                            0,
                                                            getShape().getShape()[0],
                                                            1,
                                                            DataType.INT32,
                                                            getDevice()))
                                            .asOperand(),
                                    ((TfNDArray) result).asOperand())
                            .asOutput()
                            .tensor();
            operand = null;
        }
        return this;
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
    public NDArray sum() {
        // sum on all axis
        return new TfNDArray(
                manager,
                tf.sum(asOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return new TfNDArray(
                manager, tf.sum(asOperand(), ((TfNDArray) manager.create(axes)).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        return new TfNDArray(
                manager,
                tf.prod(asOperand(), ((TfNDArray) manager.arange(0, getRank(), 1)).asOperand()));
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
        tf.splitV(
                        asOperand(),
                        tf.constant(sizes.stream().mapToLong(n -> n).toArray()),
                        tf.constant(axis),
                        (long) sizes.size())
                .forEach(output -> result.add(new TfNDArray(manager, output)));
        return null;
    }

    @Override
    public NDList split(long sections, int axis) {
        NDList result = new NDList();
        tf.split(tf.constant(axis), operand, sections)
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
        return new TfNDArray(manager, tf.math.logicalAnd(asOperand(), ((TfNDArray) n).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray n) {
        return new TfNDArray(manager, tf.math.logicalOr(asOperand(), ((TfNDArray) n).asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray n) {
        return logicalOr(n).logicalAnd(logicalAnd(n).logicalNot());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return new TfNDArray(manager, tf.math.logicalNot(asOperand()));
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
            // do transpose
            if (axis < 0) {
                axis = axis + rank;
            }

            transposition =
                    NDArrays.concat(
                            new NDList(
                                    manager.arange(axis),
                                    manager.create(new int[] {rank - 1}),
                                    manager.arange(axis + 1, rank - 1),
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
            result = topK.indices();
        } else {
            result = topK.values();
        }
        if (transposition != null) {
            result = tf.linalg.transpose(result, ((TfNDArray) transposition).asOperand());
        }
        // always return long as indices type
        return new TfNDArray(manager, tf.dtypes.cast(result, Long.class));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, float temperature) {
        return new TfNDArray(manager, tf.nn.softmax(asOperand()));
    }

    @Override
    public NDArray logSoftmax(int[] axes, float temperature) {
        return new TfNDArray(manager, tf.nn.logSoftmax(asOperand()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
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
        return new TfNDArray(manager, tf.dtypes.cast(tf.math.isInf(asOperand()), Boolean.class));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return new TfNDArray(manager, tf.dtypes.cast(tf.math.isNan(asOperand()), Boolean.class));
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
        return new TfNDArray(
                manager, tf.linalg.matMul(asOperand(), ((TfNDArray) other).asOperand()));
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
        return argMin(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
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
