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
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.tensorflow.engine.javacpp.JavacppUtils;
import ai.djl.util.NativeResource;
import ai.djl.util.Preconditions;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.tensorflow.internal.c_api.TFE_TensorHandle;
import org.tensorflow.internal.c_api.TF_Tensor;

/** {@code TfNDArray} is the TensorFlow implementation of {@link NDArray}. */
@SuppressWarnings("PMD.UseTryWithResources")
public class TfNDArray extends NativeResource<TFE_TensorHandle> implements NDArray {

    private Shape shape;
    private Device device;
    private TfNDManager manager;
    private String name;
    private TfNDArrayEx tfNDArrayEx;
    private DataType dataType;
    private TF_Tensor tensor;

    TfNDArray(TfNDManager manager, TFE_TensorHandle handle) {
        super(handle);
        this.manager = manager;
        manager.attachInternal(getUid(), this);
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    TfNDArray(TfNDManager manager, TFE_TensorHandle handle, TF_Tensor tensor) {
        this(manager, handle);
        this.tensor = tensor;
    }

    /** {@inheritDoc} */
    @Override
    public TfNDManager getManager() {
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
            Preconditions.checkArgument(
                    getHandle() != null && !getHandle().isNull(), "Eager session has been closed");
            dataType = JavacppUtils.getDataType(getHandle());
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        if (device == null) {
            Preconditions.checkArgument(
                    getHandle() != null && !getHandle().isNull(), "Eager session has been closed");
            device = JavacppUtils.getDevice(getHandle());
        }
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            Preconditions.checkArgument(
                    getHandle() != null && !getHandle().isNull(), "Eager session has been closed");
            shape = JavacppUtils.getShape(getHandle());
        }
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
        if (device.equals(getDevice()) && !copy) {
            return this;
        } else if (device.equals(getDevice()) && copy) {
            // tensorflow toDevice don't do the copy if data is already in the same device
            return duplicate();
        }
        return new TfNDArray(
                manager, JavacppUtils.toDevice(getHandle(), manager.getEagerSession(), device));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(getDataType()) && !copy) {
            return this;
        }
        return manager.opExecutor("Cast")
                .addInput(this)
                .addParam("DstT", dataType)
                .buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public void setRequiresGradient(boolean requiresGrad) {
        throw new UnsupportedOperationException("Not implemented");
    }

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
    public NDArray stopGradient() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public String[] toStringArray(Charset charset) {
        int size = Math.toIntExact(getShape().size());
        return JavacppUtils.getString(getHandle(), size, charset);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        if (getDataType() == DataType.STRING) {
            throw new IllegalArgumentException("Please use toStringArray() for String NDArray.");
        }
        return JavacppUtils.getByteBuffer(getHandle());
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        if (getDevice().isGpu()) {
            // TODO: Implement set for GPU
            throw new UnsupportedOperationException("GPU Tensor cannot be modified after creation");
        }
        int size = Math.toIntExact(getShape().size());
        BaseNDManager.validateBufferSize(data, getDataType(), size);
        if (data instanceof ByteBuffer) {
            JavacppUtils.setByteBuffer(getHandle(), (ByteBuffer) data);
            return;
        }
        ByteBuffer buf = getManager().allocateDirect(size * getDataType().getNumOfBytes());
        BaseNDManager.copyBuffer(data, buf);
        JavacppUtils.setByteBuffer(getHandle(), buf);
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (TfNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (TfNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = TfNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray ndArray) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray duplicate() {
        return manager.opExecutor("DeepCopy").addInput(this).buildSingletonOrThrow();
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
        try (NDArray where = manager.opExecutor("Where").addInput(index).buildSingletonOrThrow();
                NDArray squeeze = where.squeeze(1);
                NDArray axisArr = manager.create(axis)) {
            return manager.opExecutor("GatherV2")
                    .addInput(this)
                    .addInput(squeeze)
                    .addInput(axisArr)
                    .buildSingletonOrThrow();
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
        return manager.opExecutor("Equal").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("NotEqual")
                .addInput(this)
                .addInput(other)
                .buildSingletonOrThrow();
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
        return manager.opExecutor("Greater").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("GreaterEqual")
                .addInput(this)
                .addInput(other)
                .buildSingletonOrThrow();
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
        return manager.opExecutor("Less").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("LessEqual")
                .addInput(this)
                .addInput(other)
                .buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray all() {
        try (NDArray casted = toType(DataType.BOOLEAN, true);
                NDArray axes = manager.arange(getShape().dimension())) {
            return manager.opExecutor("All")
                    .addInput(casted)
                    .addInput(axes)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray any() {
        try (NDArray casted = toType(DataType.BOOLEAN, true);
                NDArray axes = manager.arange(getShape().dimension())) {
            return manager.opExecutor("Any")
                    .addInput(casted)
                    .addInput(axes)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray erfinv() {
        return manager.opExecutor("Erfinv").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(boolean keepDims) {
        // We have to flatten first to be able to simulate "numpy.linalg.norm" whenever axis isn't
        // specified
        if (getDataType() == DataType.FLOAT64) {
            throw new UnsupportedOperationException("float64 is not supported");
        }
        NDArray flatten = flatten();
        NDArray axis = manager.create(0);
        NDArray res =
                manager.opExecutor("EuclideanNorm")
                        .addInput(flatten)
                        .addInput(axis)
                        .addParam("keep_dims", keepDims)
                        .buildSingletonOrThrow();
        // close the temp NDArray
        flatten.close();
        axis.close();
        if (!keepDims) {
            return res;
        }
        try {
            long[] shapes = LongStream.generate(() -> 1).limit(getShape().dimension()).toArray();
            return res.reshape(shapes);
        } finally {
            res.close();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(int ord, int[] axes, boolean keepDims) {
        if (ord != 2) {
            throw new UnsupportedOperationException("Only ord=2 is supported");
        }
        try (NDArray axesArr = manager.create(axes)) {
            return manager.opExecutor("EuclideanNorm")
                    .addInput(this)
                    .addInput(axesArr)
                    .addParam("keep_dims", keepDims)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        try (NDArray depthArr = manager.create(depth);
                NDArray onValueArr = manager.create(onValue);
                NDArray offValueArr = manager.create(offValue);
                NDArray result =
                        manager.opExecutor("OneHot")
                                .addInput(this)
                                .addInput(depthArr)
                                .addInput(onValueArr)
                                .addInput(offValueArr)
                                .addParam("axis", -1)
                                .buildSingletonOrThrow()) {
            return result.toType(dataType, true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray batchDot(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
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
        return manager.opExecutor("Add").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("Sub").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("Mul").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("Div").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("Mod").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("Pow").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("Maximum").addInput(this).addInput(other).buildSingletonOrThrow();
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
        return manager.opExecutor("Minimum").addInput(this).addInput(other).buildSingletonOrThrow();
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
        TFE_TensorHandle newHandle =
                manager.opExecutor("Add").addInput(this).addInput(other).buildRawPointer(1)[0];
        setHandle(newHandle);
        return this;
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
        TFE_TensorHandle newHandle =
                manager.opExecutor("Sub").addInput(this).addInput(other).buildRawPointer(1)[0];
        setHandle(newHandle);
        return this;
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
        TFE_TensorHandle newHandle =
                manager.opExecutor("Mul").addInput(this).addInput(other).buildRawPointer(1)[0];
        setHandle(newHandle);
        return this;
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
        TFE_TensorHandle newHandle =
                manager.opExecutor("Div").addInput(this).addInput(other).buildRawPointer(1)[0];
        setHandle(newHandle);
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
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return modi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        TFE_TensorHandle newHandle =
                manager.opExecutor("Mod").addInput(this).addInput(other).buildRawPointer(1)[0];
        setHandle(newHandle);
        return this;
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
        TFE_TensorHandle newHandle =
                manager.opExecutor("Pow").addInput(this).addInput(other).buildRawPointer(1)[0];
        setHandle(newHandle);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sign() {
        return manager.opExecutor("Sign").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray signi() {
        TFE_TensorHandle newHandle =
                manager.opExecutor("Sign").addInput(this).buildRawPointer(1)[0];
        setHandle(newHandle);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return manager.opExecutor("Neg").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        TFE_TensorHandle newHandle = manager.opExecutor("Neg").addInput(this).buildRawPointer(1)[0];
        setHandle(newHandle);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return manager.opExecutor("Abs").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return manager.opExecutor("Square").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        return manager.opExecutor("Sqrt").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        NDArray pow;
        if (getDataType().equals(DataType.FLOAT64)) {
            pow = manager.create(1.0 / 3);
        } else {
            pow = manager.create(1f / 3);
        }
        try {
            return manager.opExecutor("Pow").addInput(this).addInput(pow).buildSingletonOrThrow();
        } finally {
            pow.close();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return manager.opExecutor("Floor").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return manager.opExecutor("Ceil").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return manager.opExecutor("Round").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return manager.opExecutor("Exp").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return manager.opExecutor("Log").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return log().div(Math.log(10));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return log().div(Math.log(2));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return manager.opExecutor("Sin").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return manager.opExecutor("Cos").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return manager.opExecutor("Tan").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return manager.opExecutor("Asin").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return manager.opExecutor("Acos").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return manager.opExecutor("Atan").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return manager.opExecutor("Sinh").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return manager.opExecutor("Cosh").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return manager.opExecutor("Tanh").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return manager.opExecutor("Asinh").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return manager.opExecutor("Acosh").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return manager.opExecutor("Atanh").addInput(this).buildSingletonOrThrow();
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
        try (NDArray axes = manager.arange(getShape().dimension())) {
            return manager.opExecutor("Max").addInput(this).addInput(axes).buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        try (NDArray axesArr = manager.create(axes)) {
            return manager.opExecutor("Max")
                    .addInput(this)
                    .addInput(axesArr)
                    .addParam("keep_dims", keepDims)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        try (NDArray axes = manager.arange(getShape().dimension())) {
            return manager.opExecutor("Min").addInput(this).addInput(axes).buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        try (NDArray axesArr = manager.create(axes)) {
            return manager.opExecutor("Min")
                    .addInput(this)
                    .addInput(axesArr)
                    .addParam("keep_dims", keepDims)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        // sum on all axis
        NDArray array = this;
        // tf can't sum boolean values
        if (getDataType() == DataType.BOOLEAN) {
            array = array.toType(DataType.INT64, false);
        }
        try (NDArray axes = manager.arange(getShape().dimension())) {
            return manager.opExecutor("Sum").addInput(array).addInput(axes).buildSingletonOrThrow();
        } finally {
            if (array != this) {
                array.close();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        try (NDArray axesArr = manager.create(axes)) {
            return manager.opExecutor("Sum")
                    .addInput(this)
                    .addInput(axesArr)
                    .addParam("keep_dims", keepDims)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        try (NDArray axes = manager.arange(getShape().dimension())) {
            return manager.opExecutor("Prod").addInput(this).addInput(axes).buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        try (NDArray axesArr = manager.create(axes)) {
            return manager.opExecutor("Prod")
                    .addInput(this)
                    .addInput(axesArr)
                    .addParam("keep_dims", keepDims)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        try (NDArray axes = manager.arange(getShape().dimension())) {
            return manager.opExecutor("Mean").addInput(this).addInput(axes).buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        try (NDArray axesArr = manager.create(axes)) {
            return manager.opExecutor("Mean")
                    .addInput(this)
                    .addInput(axesArr)
                    .addParam("keep_dims", keepDims)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rotate90(int times, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        if (indices.length == 0) {
            return new NDList(duplicate());
        }
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
        try (NDArray sizesArr = manager.create(sizes.stream().mapToInt(Long::intValue).toArray());
                NDArray axisArr = manager.create(axis)) {
            return new NDList(
                    manager.opExecutor("SplitV")
                            .addInput(this)
                            .addInput(sizesArr)
                            .addInput(axisArr)
                            .addParam("num_split", sizes.size())
                            .build(indices.length + 1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return reshape(new Shape(-1));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        try (NDArray shapeArr = manager.create(shape.getShape())) {
            return manager.opExecutor("Reshape")
                    .addInput(this)
                    .addInput(shapeArr)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        try (NDArray axisArr = manager.create(axis)) {
            return manager.opExecutor("ExpandDims")
                    .addInput(this)
                    .addInput(axisArr)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        if (isScalar()) {
            axes = new int[0];
        }
        return manager.opExecutor("Squeeze")
                .addInput(this)
                .addParam("squeeze_dims", Arrays.stream(axes).mapToLong(i -> i).toArray())
                .buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray n) {
        try (NDArray input1 = toType(DataType.BOOLEAN, true);
                NDArray input2 = n.toType(DataType.BOOLEAN, true)) {
            return manager.opExecutor("LogicalAnd")
                    .addInput(input1)
                    .addInput(input2)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray n) {
        try (NDArray input1 = toType(DataType.BOOLEAN, true);
                NDArray input2 = n.toType(DataType.BOOLEAN, true)) {
            return manager.opExecutor("LogicalOr")
                    .addInput(input1)
                    .addInput(input2)
                    .buildSingletonOrThrow();
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
        try (NDArray input = toType(DataType.BOOLEAN, true)) {
            return manager.opExecutor("LogicalNot").addInput(input).buildSingletonOrThrow();
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

    private NDArray sortHelper(int axis, boolean ascending, boolean returnIndices) {
        if (isScalar()) {
            return duplicate();
        }
        // using topK to implement argSort
        int k;
        int rank = getShape().dimension();
        int[] transposition;
        NDArray input;
        NDArray result;
        try (TfNDManager subManager = (TfNDManager) manager.newSubManager()) {
            attach(subManager);
            if (axis == -1 || axis + 1 == getShape().dimension()) {
                // last axis
                transposition = null;
                input = this;
                long[] arrayShape = getShape().getShape();
                k = (int) arrayShape[arrayShape.length - 1];
            } else {
                k = (int) getShape().getShape()[axis];
                NDArray axisArr = subManager.arange(0, axis, 1, DataType.INT32, getDevice());
                NDArray axisArr1 = subManager.create(new int[] {rank - 1});
                NDArray axisArr2 =
                        subManager.arange(axis + 1, rank - 1, 1, DataType.INT32, getDevice());
                NDArray axisArr3 = subManager.create(new int[] {axis});
                transposition =
                        NDArrays.concat(new NDList(axisArr, axisArr1, axisArr2, axisArr3))
                                .toIntArray();
                input = transpose(transposition);
            }
            NDArray[] outputs;
            NDArray kArr = subManager.create(k);
            if (ascending) {
                outputs =
                        subManager
                                .opExecutor("TopKV2")
                                .addInput(input.neg())
                                .addInput(kArr)
                                .build(2);
            } else {
                outputs = subManager.opExecutor("TopKV2").addInput(input).addInput(kArr).build(2);
            }
            if (returnIndices) {
                result = outputs[1].toType(DataType.INT64, false);
            } else {
                result = outputs[0];
            }
            if (transposition != null) {
                result = result.transpose(transposition);
            }
            // re-apply neg after sort if ascending
            if (ascending && !returnIndices) {
                result = result.neg();
            }
            attach(subManager.getParentManager());
            result.attach(subManager.getParentManager());
            return result;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int axis) {
        return softmaxHelper("Softmax", axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int axis) {
        return softmaxHelper("LogSoftmax", axis);
    }

    private NDArray softmaxHelper(String operation, int axis) {
        long dim = getShape().dimension();
        if (dim == 0) {
            return duplicate();
        }
        if (axis == -1 || axis == dim - 1) {
            return manager.opExecutor(operation).addInput(this).buildSingletonOrThrow();
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
        try (NDManager subManager = manager.newSubManager()) {
            attach(subManager);
            NDList concatList = new NDList();
            concatList.add(subManager.arange((int) (axis % dim)));
            concatList.add(subManager.create((int) dim - 1).expandDims(0));
            concatList.add(subManager.arange(axis + 1, (int) dim - 1));
            concatList.add(subManager.create(axis).expandDims(0));
            int[] axes = NDArrays.concat(concatList, 0).toIntArray();
            NDArray transposed = transpose(axes);
            NDArray output =
                    ((TfNDManager) subManager)
                            .opExecutor(operation)
                            .addInput(transposed)
                            .buildSingletonOrThrow();
            NDArray result = output.transpose(axes);

            result.attach(subManager.getParentManager());
            attach(subManager.getParentManager());
            return result;
        }
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
        try (NDArray axisArr = manager.create(axis)) {
            return manager.opExecutor("Cumsum")
                    .addInput(this)
                    .addInput(axisArr)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return cumSum(0);
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return manager.opExecutor("IsInf").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return manager.opExecutor("IsNan").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        // tf tile doesn't support scalar
        if (isScalar()) {
            try (NDArray temp = reshape(1)) {
                return temp.tile(repeats);
            }
        }
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
        try (NDArray repeatsArr = manager.create(repeats)) {
            return manager.opExecutor("Tile")
                    .addInput(this)
                    .addInput(repeatsArr)
                    .buildSingletonOrThrow();
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
    public NDArray matMul(NDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for matMul()");
        }
        if (getShape().dimension() > 2 || other.getShape().dimension() > 2) {
            return manager.opExecutor("BatchMatMulV2")
                    .addInput(this)
                    .addInput(other)
                    .buildSingletonOrThrow();
        }
        boolean broadcast = false;
        NDArray lhs = this;
        NDArray rhs = other;
        if (getShape().dimension() == 1) {
            lhs = broadcast(1, getShape().get(0));
            broadcast = true;
        }
        if (other.getShape().dimension() == 1) {
            rhs = rhs.broadcast(1, getShape().get(0));
            broadcast = true;
        }
        NDArray result =
                manager.opExecutor("MatMul").addInput(lhs).addInput(rhs).buildSingletonOrThrow();
        try {
            if (broadcast) {
                return result.squeeze();
            }
            return result;
        } finally {
            if (lhs != this) {
                lhs.close();
            }
            if (rhs != other) {
                rhs.close();
            }
            if (broadcast) {
                result.close();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        try (NDArray minArr = manager.create(min.floatValue());
                NDArray maxArr = manager.create(max.floatValue())) {
            return manager.opExecutor("ClipByValue")
                    .addInput(this)
                    .addInput(minArr)
                    .addInput(maxArr)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flip(int... axes) {
        try (NDArray axesArr = manager.create(axes)) {
            return manager.opExecutor("ReverseV2")
                    .addInput(this)
                    .addInput(axesArr)
                    .buildSingletonOrThrow();
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
        try (NDArray dimensionsArr = manager.create(dimensions)) {
            return manager.opExecutor("Transpose")
                    .addInput(this)
                    .addInput(dimensionsArr)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        try (NDArray shapeArr = manager.create(shape.getShape())) {
            return manager.opExecutor("BroadcastTo")
                    .addInput(this)
                    .addInput(shapeArr)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty NDArray");
        }
        return manager.opExecutor("ArgMax").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        try (NDArray axisArr = manager.create(axis)) {
            return manager.opExecutor("ArgMax")
                    .addInput(this)
                    .addInput(axisArr)
                    .buildSingletonOrThrow();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty NDArray");
        }
        return manager.opExecutor("ArgMin").addInput(this).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        try (NDArray axisArr = manager.create(axis)) {
            return manager.opExecutor("ArgMin")
                    .addInput(this)
                    .addInput(axisArr)
                    .buildSingletonOrThrow();
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
        if (isReleased()) {
            return "This array is already closed";
        }
        return toDebugString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        TFE_TensorHandle tensorHandle = handle.getAndSet(null);
        if (tensorHandle != null && !tensorHandle.isNull()) {
            tensorHandle.close();
            if (tensor != null) {
                tensor.close();
            }
            manager.detachInternal(getUid());
            manager = null;
        }
        tfNDArrayEx = null;
    }

    // TensorFlow doesn't support in-place operation
    // each operator execution will generate a new node in the graph
    // workaround the limitation by updating the handle
    protected void setHandle(TFE_TensorHandle newHandle) {
        TFE_TensorHandle oldHandle = handle.getAndSet(newHandle);
        oldHandle.close();
    }
}
