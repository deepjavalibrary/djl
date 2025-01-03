/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.function.Function;

/**
 * A base implementation of the {@link NDArray} that does nothing. This can be used for overriding
 * the NDArray with only part of the interface implemented.
 *
 * <p>This interface should only be used for the NDArray implementations that do not plan to
 * implement a large portion of the interface. For the ones that do, they should directly implement
 * {@link NDArray} so that the unsupported operations are better highlighted in the code.
 */
public abstract class NDArrayAdapter implements NDArray {

    private static final String UNSUPPORTED_MSG =
            "This NDArray implementation does not currently support this operation";

    protected NDManager manager;
    protected NDManager alternativeManager;
    protected NDArray alternativeArray;

    protected Shape shape;
    protected DataType dataType;
    protected String name;
    protected boolean isClosed;
    protected String uid;

    protected NDArrayAdapter(
            NDManager manager,
            NDManager alternativeManager,
            Shape shape,
            DataType dataType,
            String uid) {
        this.manager = manager;
        this.alternativeManager = alternativeManager;
        this.shape = shape;
        this.dataType = dataType;
        this.uid = uid;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = manager;
        manager.attachInternal(getUid(), this);
        alternativeManager = ((BaseNDManager) manager).getAlternativeManager();
        if (alternativeManager == null) {
            // to prevent hybrid engine memory leak
            alternativeManager = manager;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        NDManager original = this.manager;
        detach();
        this.manager = manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return SparseFormat.DENSE;
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
    public Device getDevice() {
        return manager.getDevice();
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (isClosed) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (isClosed) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        if (device.equals(getDevice())) {
            if (copy) {
                return duplicate();
            }
            return this;
        }
        NDArray array = getManager().create(getShape(), getDataType(), device);
        array.setName(getName());
        copyTo(array);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(getDataType())) {
            if (copy) {
                return duplicate();
            }
            return this;
        }
        Number[] numbers = toArray();
        ByteBuffer bb = toTypeInternal(numbers, dataType);
        NDArray array = manager.create(bb, getShape(), dataType);
        array.setName(getName());
        return array;
    }

    private ByteBuffer toTypeInternal(Number[] numbers, DataType dataType) {
        int size = dataType.getNumOfBytes() * numbers.length;
        ByteBuffer bb = manager.allocateDirect(size);
        for (Number number : numbers) {
            switch (dataType) {
                case FLOAT16:
                case FLOAT32:
                    bb.putFloat(number.floatValue());
                    break;
                case FLOAT64:
                    bb.putDouble(number.doubleValue());
                    break;
                case INT16:
                case UINT16:
                    bb.putShort(number.shortValue());
                    break;
                case INT32:
                case UINT32:
                    bb.putInt(number.intValue());
                    break;
                case INT64:
                case UINT64:
                    bb.putLong(number.longValue());
                    break;
                case BOOLEAN:
                case INT8:
                case UINT8:
                    bb.put(number.byteValue());
                    break;
                default:
                    throw new IllegalStateException("Unsupported DataType: " + getDataType());
            }
        }
        bb.rewind();
        return bb;
    }

    /** {@inheritDoc} */
    @Override
    public void setRequiresGradient(boolean requiresGrad) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasGradient() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stopGradient() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public String[] toStringArray(Charset charset) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gather(NDArray index, int axis) {
        return getAlternativeArray().gather(alternativeManager.from(index), axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gatherNd(NDArray index) {
        return getAlternativeArray().gatherNd(alternativeManager.from(index));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray take(NDManager manager, NDArray index) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(NDArray index, NDArray value) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray scatter(NDArray index, NDArray value, int axis) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        return get(alternativeManager, index);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer buffer) {
        NDArray array = manager.create(buffer, getShape(), getDataType());
        intern(array);
        array.detach();
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, NDArray value) {
        getAlternativeArray().set(index, value);
        set(alternativeArray.toByteBuffer());
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Number value) {
        getAlternativeArray().set(index, value);
        set(alternativeArray.toByteBuffer());
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Function<NDArray, NDArray> function) {
        getAlternativeArray().set(index, function);
        set(alternativeArray.toByteBuffer());
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray index, Number value) {
        getAlternativeArray().set(index, value);
        set(alternativeArray.toByteBuffer());
    }

    /** {@inheritDoc} */
    @Override
    public void setScalar(NDIndex index, Number value) {
        getAlternativeArray().setScalar(index, value);
        set(alternativeArray.toByteBuffer());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        return getAlternativeArray().booleanMask(alternativeManager.from(index), axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        return getAlternativeArray().sequenceMask(alternativeManager.from(sequenceLength), value);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength) {
        return sequenceMask(sequenceLength, 0);
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return Arrays.stream(toArray()).allMatch(n -> n.equals(number));
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        if (other instanceof NDArrayAdapter) {
            return getShape().equals(other.getShape())
                    && Arrays.equals(toByteArray(), other.toByteArray());
        }
        return other.contentEquals(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number n) {
        return getAlternativeArray().eq(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return getAlternativeArray().eq(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number n) {
        return getAlternativeArray().neq(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return getAlternativeArray().neq(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number n) {
        return getAlternativeArray().gt(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return getAlternativeArray().gt(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number n) {
        return getAlternativeArray().gte(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return getAlternativeArray().gte(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number n) {
        return getAlternativeArray().lt(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return getAlternativeArray().lt(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number n) {
        return getAlternativeArray().lte(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return getAlternativeArray().lte(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return getAlternativeArray().add(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return getAlternativeArray().add(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return getAlternativeArray().sub(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return getAlternativeArray().sub(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return getAlternativeArray().mul(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return getAlternativeArray().mul(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return getAlternativeArray().div(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return getAlternativeArray().div(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return getAlternativeArray().mod(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return getAlternativeArray().mod(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        return getAlternativeArray().pow(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        return getAlternativeArray().pow(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sign() {
        return getAlternativeArray().sign();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray signi() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        return getAlternativeArray().maximum(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        return getAlternativeArray().maximum(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        return getAlternativeArray().minimum(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        return getAlternativeArray().minimum(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return getAlternativeArray().neg();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return getAlternativeArray().abs();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return getAlternativeArray().square();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        return getAlternativeArray().sqrt();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return getAlternativeArray().cbrt();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return getAlternativeArray().floor();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return getAlternativeArray().ceil();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return getAlternativeArray().round();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return getAlternativeArray().trunc();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return getAlternativeArray().exp();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gammaln() {
        return getAlternativeArray().gammaln();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return getAlternativeArray().log();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return getAlternativeArray().log10();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return getAlternativeArray().log2();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return getAlternativeArray().sin();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return getAlternativeArray().cos();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return getAlternativeArray().tan();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return getAlternativeArray().asin();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return getAlternativeArray().acos();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return getAlternativeArray().atan();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan2(NDArray other) {
        return getAlternativeArray().atan2(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return getAlternativeArray().sinh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return getAlternativeArray().cosh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return getAlternativeArray().tanh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return getAlternativeArray().asinh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return getAlternativeArray().acosh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return getAlternativeArray().atanh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return getAlternativeArray().toDegrees();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return getAlternativeArray().toRadians();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        return getAlternativeArray().max();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return getAlternativeArray().max(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        return getAlternativeArray().min();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return getAlternativeArray().min(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        return getAlternativeArray().sum();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return getAlternativeArray().sum(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumProd(int axis) {
        return getAlternativeArray().cumProd(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumProd(int axis, DataType dataType) {
        return getAlternativeArray().cumProd(axis, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        return getAlternativeArray().prod();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return getAlternativeArray().prod(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return getAlternativeArray().mean();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return getAlternativeArray().mean(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray normalize(double p, long dim, double eps) {
        return getAlternativeArray().normalize(p, dim, eps);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rotate90(int times, int[] axes) {
        return getAlternativeArray().rotate90(times, axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return getAlternativeArray().trace(offset, axis1, axis2);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long sections, int axis) {
        return getAlternativeArray().split(sections, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        return getAlternativeArray().split(indices, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return getAlternativeArray().flatten();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten(int startDim, int endDim) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ifft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rfft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray irfft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stft(
            long nFft,
            long hopLength,
            boolean center,
            NDArray window,
            boolean normalize,
            boolean returnComplex) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fft2(long[] sizes, long[] axes) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pad(Shape padding, double value) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ifft2(long[] sizes, long[] axes) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return getAlternativeArray().reshape(shape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        return getAlternativeArray().expandDims(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        return getAlternativeArray().squeeze(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDList unique(Integer dim, boolean sorted, boolean returnInverse, boolean returnCounts) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray other) {
        return getAlternativeArray().logicalAnd(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        return getAlternativeArray().logicalOr(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        return getAlternativeArray().logicalXor(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return getAlternativeArray().logicalNot();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return getAlternativeArray().argSort(axis, ascending);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return getAlternativeArray().sort();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return getAlternativeArray().sort(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int axis) {
        return getAlternativeArray().softmax(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int axis) {
        return getAlternativeArray().logSoftmax(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return getAlternativeArray().cumSum();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        return getAlternativeArray().cumSum(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return getAlternativeArray().isInfinite();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return getAlternativeArray().isNaN();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        return getAlternativeArray().tile(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        return getAlternativeArray().tile(axis, repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        return getAlternativeArray().tile(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return getAlternativeArray().tile(desiredShape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        return getAlternativeArray().repeat(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        return getAlternativeArray().repeat(axis, repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        return getAlternativeArray().repeat(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return getAlternativeArray().repeat(desiredShape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        return getAlternativeArray().dot(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        return getAlternativeArray().matMul(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray batchMatMul(NDArray other) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray xlogy(NDArray other) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        return getAlternativeArray().clip(min, max);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flip(int... axes) {
        return getAlternativeArray().flip(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return getAlternativeArray().transpose();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... axes) {
        return getAlternativeArray().transpose(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        return getAlternativeArray().broadcast(shape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        return getAlternativeArray().argMax();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        return getAlternativeArray().argMax(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDList topK(int k, int axis, boolean largest, boolean sorted) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        return getAlternativeArray().argMin();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        return getAlternativeArray().argMin(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        return getAlternativeArray().percentile(percentile);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] axes) {
        return getAlternativeArray().percentile(percentile, axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        return getAlternativeArray().median();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        return getAlternativeArray().median(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        return getAlternativeArray().toDense();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return getAlternativeArray().toSparse(fmt);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        return getAlternativeArray().nonzero();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray erfinv() {
        return getAlternativeArray().erfinv();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray erf() {
        return getAlternativeArray().erf();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray inverse() {
        return getAlternativeArray().inverse();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(boolean keepDims) {
        return getAlternativeArray().norm(keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(int ord, int[] axes, boolean keepDims) {
        return getAlternativeArray().norm(ord, axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        return getAlternativeArray().oneHot(depth, onValue, offValue, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray batchDot(NDArray other) {
        return getAlternativeArray().batchDot(alternativeManager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray complex() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray real() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray conj() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        NDArray array = getAlternativeArray();
        if (array instanceof NDArrayAdapter) {
            throw new UnsupportedOperationException("Operation not supported.");
        }
        return array.getNDArrayInternal();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isReleased() {
        return isClosed;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!isClosed) {
            manager.detachInternal(getUid());
            isClosed = true;
            if (alternativeArray != null) {
                alternativeArray.close();
                alternativeArray = null;
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof NDArray) {
            return contentEquals((NDArray) obj);
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
        if (isClosed) {
            return "This array is already closed";
        }
        return toDebugString();
    }

    private NDArray getAlternativeArray() {
        if (alternativeManager == null) {
            throw new UnsupportedOperationException(UNSUPPORTED_MSG);
        }
        if (alternativeArray == null) {
            alternativeArray = alternativeManager.from(this);
        } else {
            alternativeArray.set(getDataType().asDataType(toByteBuffer()));
        }
        NDScope.unregister(alternativeArray);
        return alternativeArray;
    }
}
