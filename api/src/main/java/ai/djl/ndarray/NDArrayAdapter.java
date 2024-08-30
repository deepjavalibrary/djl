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
    private NDArray alternativeArray;

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
        return correctedArray(getAlternativeArray().gather(alternativeManager.from(index), axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gatherNd(NDArray index) {
        return correctedArray(getAlternativeArray().gatherNd(alternativeManager.from(index)));
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
        return correctedArray(get(alternativeManager, index));
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
        return correctedArray(getAlternativeArray().booleanMask(alternativeManager.from(index), axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        return correctedArray(getAlternativeArray().sequenceMask(alternativeManager.from(sequenceLength), value));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength) {
        return correctedArray(sequenceMask(sequenceLength, 0));
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
        return correctedArray(getAlternativeArray().eq(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return correctedArray(getAlternativeArray().eq(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number n) {
        return correctedArray(getAlternativeArray().neq(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return correctedArray(getAlternativeArray().neq(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number n) {
        return correctedArray(getAlternativeArray().gt(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return correctedArray(getAlternativeArray().gt(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number n) {
        return correctedArray(getAlternativeArray().gte(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return correctedArray(getAlternativeArray().gte(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number n) {
        return correctedArray(getAlternativeArray().lt(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return correctedArray(getAlternativeArray().lt(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number n) {
        return correctedArray(getAlternativeArray().lte(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return correctedArray(getAlternativeArray().lte(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return correctedArray(getAlternativeArray().add(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return correctedArray(getAlternativeArray().add(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return correctedArray(getAlternativeArray().sub(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return correctedArray(getAlternativeArray().sub(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return correctedArray(getAlternativeArray().mul(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return correctedArray(getAlternativeArray().mul(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return correctedArray(getAlternativeArray().div(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return correctedArray(getAlternativeArray().div(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return correctedArray(getAlternativeArray().mod(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return correctedArray(getAlternativeArray().mod(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        return correctedArray(getAlternativeArray().pow(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        return correctedArray(getAlternativeArray().pow(alternativeManager.from(other)));
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
        return correctedArray(getAlternativeArray().sign());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray signi() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        return correctedArray(getAlternativeArray().maximum(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        return correctedArray(getAlternativeArray().maximum(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        return correctedArray(getAlternativeArray().minimum(n));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        return correctedArray(getAlternativeArray().minimum(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return correctedArray(getAlternativeArray().neg());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return correctedArray(getAlternativeArray().abs());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return correctedArray(getAlternativeArray().square());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        return correctedArray(getAlternativeArray().sqrt());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return correctedArray(getAlternativeArray().cbrt());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return correctedArray(getAlternativeArray().floor());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return correctedArray(getAlternativeArray().ceil());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return correctedArray(getAlternativeArray().round());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return correctedArray(getAlternativeArray().trunc());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return correctedArray(getAlternativeArray().exp());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gammaln() {
        return correctedArray(getAlternativeArray().gammaln());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return correctedArray(getAlternativeArray().log());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return correctedArray(getAlternativeArray().log10());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return correctedArray(getAlternativeArray().log2());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return correctedArray(getAlternativeArray().sin());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return correctedArray(getAlternativeArray().cos());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return correctedArray(getAlternativeArray().tan());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return correctedArray(getAlternativeArray().asin());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return correctedArray(getAlternativeArray().acos());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return correctedArray(getAlternativeArray().atan());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan2(NDArray other) {
        return correctedArray(getAlternativeArray().atan2(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return correctedArray(getAlternativeArray().sinh());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return correctedArray(getAlternativeArray().cosh());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return correctedArray(getAlternativeArray().tanh());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return correctedArray(getAlternativeArray().asinh());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return correctedArray(getAlternativeArray().acosh());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return correctedArray(getAlternativeArray().atanh());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return correctedArray(getAlternativeArray().toDegrees());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return correctedArray(getAlternativeArray().toRadians());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        return correctedArray(getAlternativeArray().max());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return correctedArray(getAlternativeArray().max(axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        return correctedArray(getAlternativeArray().min());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return correctedArray(getAlternativeArray().min(axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        return correctedArray(getAlternativeArray().sum());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return correctedArray(getAlternativeArray().sum(axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumProd(int axis) {
        return correctedArray(getAlternativeArray().cumProd(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumProd(int axis, DataType dataType) {
        return correctedArray(getAlternativeArray().cumProd(axis, dataType));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        return correctedArray(getAlternativeArray().prod());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return correctedArray(getAlternativeArray().prod(axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return correctedArray(getAlternativeArray().mean());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return correctedArray(getAlternativeArray().mean(axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray normalize(double p, long dim, double eps) {
        return correctedArray(getAlternativeArray().normalize(p, dim, eps));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rotate90(int times, int[] axes) {
        return correctedArray(getAlternativeArray().rotate90(times, axes));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return correctedArray(getAlternativeArray().trace(offset, axis1, axis2));
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long sections, int axis) {
        NDList list = getAlternativeArray().split(sections, axis);
        NDArray[] corrected = new NDArray[list.size()];
        for (int i = 0; i < list.size(); i++)
        {
            corrected[i] = correctedArray(list.get(i));
        }
        return new NDList(corrected);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        NDList list = getAlternativeArray().split(indices, axis);
        NDArray[] corrected = new NDArray[list.size()];
        for (int i = 0; i < list.size(); i++)
        {
            corrected[i] = correctedArray(list.get(i));
        }
        return new NDList(corrected);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return correctedArray(getAlternativeArray().flatten());
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
        return correctedArray(getAlternativeArray().reshape(shape));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        return correctedArray(getAlternativeArray().expandDims(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        return correctedArray(getAlternativeArray().squeeze(axes));
    }

    /** {@inheritDoc} */
    @Override
    public NDList unique(Integer dim, boolean sorted, boolean returnInverse, boolean returnCounts) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray other) {
        return correctedArray(getAlternativeArray().logicalAnd(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        return correctedArray(getAlternativeArray().logicalOr(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        return correctedArray(getAlternativeArray().logicalXor(alternativeManager.from(other)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return correctedArray(getAlternativeArray().logicalNot());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return correctedArray(getAlternativeArray().argSort(axis, ascending));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return correctedArray(getAlternativeArray().sort());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return correctedArray(getAlternativeArray().sort(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int axis) {
        return correctedArray(getAlternativeArray().softmax(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int axis) {
        return correctedArray(getAlternativeArray().logSoftmax(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return correctedArray(getAlternativeArray().cumSum());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        return correctedArray(getAlternativeArray().cumSum(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return correctedArray(getAlternativeArray().isInfinite());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return correctedArray(getAlternativeArray().isNaN());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        return correctedArray(getAlternativeArray().tile(repeats));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        return correctedArray(getAlternativeArray().tile(axis, repeats));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        return correctedArray(getAlternativeArray().tile(repeats));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return correctedArray(getAlternativeArray().tile(desiredShape));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        return correctedArray(getAlternativeArray().repeat(repeats));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        return correctedArray(getAlternativeArray().repeat(axis, repeats));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        return correctedArray(getAlternativeArray().repeat(repeats));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return correctedArray(getAlternativeArray().repeat(desiredShape));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        return correctedArray(getAlternativeArray().dot(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        return correctedArray(getAlternativeArray().matMul(other));
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
        return correctedArray(getAlternativeArray().clip(min, max));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flip(int... axes) {
        return correctedArray(getAlternativeArray().flip(axes));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return correctedArray(getAlternativeArray().transpose());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... axes) {
        return correctedArray(getAlternativeArray().transpose(axes));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        return correctedArray(getAlternativeArray().broadcast(shape));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        return correctedArray(getAlternativeArray().argMax());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        return correctedArray(getAlternativeArray().argMax(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDList topK(int k, int axis, boolean largest, boolean sorted) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        return correctedArray(getAlternativeArray().argMin());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        return correctedArray(getAlternativeArray().argMin(axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        return correctedArray(getAlternativeArray().percentile(percentile));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] axes) {
        return correctedArray(getAlternativeArray().percentile(percentile, axes));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        return correctedArray(getAlternativeArray().median());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        return correctedArray(getAlternativeArray().median(axes));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        return correctedArray(getAlternativeArray().toDense());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return correctedArray(getAlternativeArray().toSparse(fmt));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        return correctedArray(getAlternativeArray().nonzero());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray erfinv() {
        return correctedArray(getAlternativeArray().erfinv());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray erf() {
        return correctedArray(getAlternativeArray().erf());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray inverse() {
        return correctedArray(getAlternativeArray().inverse());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(boolean keepDims) {
        return correctedArray(getAlternativeArray().norm(keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(int ord, int[] axes, boolean keepDims) {
        return correctedArray(getAlternativeArray().norm(ord, axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        return correctedArray(getAlternativeArray().oneHot(depth, onValue, offValue, dataType));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray batchDot(NDArray other) {
        return correctedArray(getAlternativeArray().batchDot(alternativeManager.from(other)));
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
        return alternativeArray;
    }

    /**
     * Returns a corrected array ensuring it is owned by the main NDManager and not the alternative
     * manager. This is required for hybrid engine support.
     *
     * @param array the NDArray to return.
     * @return the NDArray to return.
     */
    private NDArray correctedArray(NDArray array)
    {
        if (array.getManager() != manager) {
            // Handle hybrid engine arrays, copy the data to a new array owned by the expected manager.
            NDArray corrected = manager.create(array.getShape(), array.getDataType(), array.getDevice());
            array.copyTo(corrected);
            corrected.setName(array.getName());

            // No need to keep the old array anymore.
            array.close();
            return corrected;
        }
        else {
            return array;
        }
    }
}
