/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rust;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.NativeResource;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.stream.IntStream;

/** {@code RsNDArray} is the Rust implementation of {@link NDArray}. */
@SuppressWarnings("try")
public class RsNDArray extends NativeResource<Long> implements NDArray {

    private String name;
    private Device device;
    private DataType dataType;
    private Shape shape;
    private RsNDManager manager;
    private RsNDArrayEx ndArrayEx;

    // keep a reference to direct buffer to avoid GC release the memory
    @SuppressWarnings("PMD.UnusedPrivateField")
    private ByteBuffer dataRef;

    /**
     * Constructs a Rust {@code NDArray} from a native handle (internal. Use {@link NDManager}
     * instead).
     *
     * @param manager the manager to attach the new array to
     * @param handle the pointer to the native Rust memory
     */
    @SuppressWarnings("this-escape")
    public RsNDArray(RsNDManager manager, long handle) {
        this(manager, handle, null, null);
    }

    @SuppressWarnings("this-escape")
    RsNDArray(RsNDManager manager, long handle, DataType dataType) {
        this(manager, handle, dataType, null);
    }

    /**
     * Constructs a Rust {@code NDArray} from a native handle (internal. Use {@link NDManager}
     * instead) with the data that is hold on Java side.
     *
     * @param manager the manager to attach the new array to
     * @param handle the pointer to the native Rust memory
     * @param dataType the {@link DataType} to be set
     * @param data the direct buffer of the data
     */
    @SuppressWarnings("this-escape")
    public RsNDArray(RsNDManager manager, long handle, DataType dataType, ByteBuffer data) {
        super(handle);
        this.dataType = dataType;
        this.manager = manager;
        this.ndArrayEx = new RsNDArrayEx(this);
        dataRef = data;
        manager.attachInternal(getUid(), this);
        NDScope.register(this);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDManager getManager() {
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
            int type = RustLibrary.getDataType(getHandle());
            dataType = DataType.values()[type];
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        if (device == null) {
            int[] dev = RustLibrary.getDevice(getHandle());
            String deviceType;
            switch (dev[0]) {
                case 0:
                    deviceType = Device.Type.CPU;
                    break;
                case 1:
                    deviceType = Device.Type.GPU;
                    break;
                case 2:
                    deviceType = "mps";
                    break;
                default:
                    throw new EngineException("Unknown device type: " + dev[0]);
            }
            device = Device.of(deviceType, dev[1]);
        }
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = new Shape(RustLibrary.getShape(getHandle()));
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return SparseFormat.DENSE;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray toDevice(Device device, boolean copy) {
        if (device.equals(getDevice()) && !copy) {
            return this;
        }
        String deviceType = device.getDeviceType();
        long newHandle = RustLibrary.toDevice(getHandle(), deviceType, device.getDeviceId());
        return toArray(newHandle, null, false, true);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(getDataType()) && !copy) {
            return this;
        }
        if (dataType == DataType.BOOLEAN) {
            long newHandle = RustLibrary.toBoolean(getHandle());
            return toArray(newHandle, dataType, false, true);
        }
        if (this.dataType == DataType.INT64
                && dataType == DataType.FLOAT16
                && getDevice().isGpu()) {
            // TODO:
            throw new UnsupportedOperationException("FP16 to I64 is not supported on GPU.");
        }
        int dType = manager.toRustDataType(dataType);
        long newHandle = RustLibrary.toDataType(getHandle(), dType);
        return toArray(newHandle, dataType, false, true);
    }

    /** {@inheritDoc} */
    @Override
    public void setRequiresGradient(boolean requiresGrad) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray getGradient() {
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
    public ByteBuffer toByteBuffer(boolean tryDirect) {
        byte[] buf = RustLibrary.toByteArray(getHandle());
        ByteBuffer bb = ByteBuffer.wrap(buf);
        bb.order(ByteOrder.nativeOrder());
        return bb;
    }

    /** {@inheritDoc} */
    @Override
    public String[] toStringArray(Charset charset) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer buffer) {
        int size = Math.toIntExact(size());
        DataType type = getDataType();
        BaseNDManager.validateBuffer(buffer, type, size);
        // TODO how do we handle the exception happened in the middle
        dataRef = null;
        if (buffer.isDirect() && buffer instanceof ByteBuffer) {
            // If NDArray is on the GPU, it is native code responsibility to control the data life
            // cycle
            if (!getDevice().isGpu()) {
                dataRef = (ByteBuffer) buffer;
            }
            intern(manager.create(buffer, getShape(), type).toDevice(getDevice(), false));
            return;
        }
        // int8, uint8, boolean use ByteBuffer, so need to explicitly input DataType
        ByteBuffer buf = manager.allocateDirect(size * type.getNumOfBytes());
        BaseNDManager.copyBuffer(buffer, buf);

        // If NDArray is on the GPU, it is native code responsibility to control the data life cycle
        if (!getDevice().isGpu()) {
            dataRef = buf;
        }
        intern(manager.create(buf, getShape(), type).toDevice(getDevice(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gather(NDArray index, int axis) {
        //        try (NDScope ignore = new NDScope()) {
        //            long indexHandle = manager.from(index).getHandle();
        //            return toArray(RustLibrary.gather(getHandle(), indexHandle, axis), true);
        //        }
        // TODO:
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gatherNd(NDArray index) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray take(NDManager manager, NDArray index) {
        try (NDScope ignore = new NDScope()) {
            long indexHandle = this.manager.from(index).getHandle();
            long newHandle = RustLibrary.take(getHandle(), indexHandle);
            RsNDArray array = new RsNDArray((RsNDManager) manager, newHandle);
            NDScope.unregister(array);
            return array;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(NDArray index, NDArray value) {
        try (NDScope ignore = new NDScope()) {
            long indexHandle = manager.from(index).getHandle();
            long valueHandle = manager.from(value).getHandle();
            return toArray(RustLibrary.put(getHandle(), indexHandle, valueHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray scatter(NDArray index, NDArray value, int axis) {
        //        try (NDScope ignore = new NDScope()) {
        //            long indexHandle = manager.from(index).getHandle();
        //            long valueHandle = manager.from(value).getHandle();
        //            return toArray(RustLibrary.scatter(getHandle(), indexHandle, valueHandle,
        // axis), true);
        //        }
        // TODO:
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (RsNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void returnResource(NDManager manager) {
        detach();
        this.manager = (RsNDManager) manager;
        manager.attachUncappedInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        NDManager original = this.manager;
        detach();
        this.manager = (RsNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = RsNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray duplicate() {
        return toArray(RustLibrary.duplicate(getHandle()), dataType, false, true);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray booleanMask(NDArray index, int axis) {
        throw new UnsupportedOperationException("Not implemented");
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
        return contentEquals(manager.create(number));
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
        return RustLibrary.contentEqual(getHandle(), manager.from(other).getHandle());
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray eq(Number n) {
        try (NDArray number = manager.create(n)) {
            return eq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray eq(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long newHandle = RustLibrary.eq(getHandle(), manager.from(other).getHandle());
            return toArray(newHandle, DataType.BOOLEAN, true, false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray neq(Number n) {
        try (NDArray number = manager.create(n)) {
            return neq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray neq(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long newHandle = RustLibrary.neq(getHandle(), manager.from(other).getHandle());
            return toArray(newHandle, DataType.BOOLEAN, true, false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray gt(Number n) {
        try (NDArray number = manager.create(n)) {
            return gt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray gt(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long newHandle = RustLibrary.gt(getHandle(), manager.from(other).getHandle());
            return toArray(newHandle, DataType.BOOLEAN, true, false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray gte(Number n) {
        try (NDArray number = manager.create(n)) {
            return gte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray gte(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long newHandle = RustLibrary.gte(getHandle(), manager.from(other).getHandle());
            return toArray(newHandle, DataType.BOOLEAN, true, false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray lt(Number n) {
        try (NDArray number = manager.create(n)) {
            return lt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray lt(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long newHandle = RustLibrary.lt(getHandle(), manager.from(other).getHandle());
            return toArray(newHandle, DataType.BOOLEAN, true, false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray lte(Number n) {
        try (NDArray number = manager.create(n)) {
            return lte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray lte(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long newHandle = RustLibrary.lte(getHandle(), manager.from(other).getHandle());
            return toArray(newHandle, DataType.BOOLEAN, true, false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray add(Number n) {
        try (NDArray number = manager.create(n)) {
            return add(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray add(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.add(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sub(Number n) {
        try (NDArray number = manager.create(n)) {
            return sub(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sub(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.sub(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray mul(Number n) {
        try (NDArray number = manager.create(n)) {
            return mul(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray mul(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.mul(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray div(Number n) {
        try (NDArray number = manager.create(n)) {
            return div(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray div(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.div(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray mod(Number n) {
        try (NDArray number = manager.create(n)) {
            return mod(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray mod(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long otherHandle = manager.from(other).getHandle();
            return toArray(RustLibrary.remainder(getHandle(), otherHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray pow(Number n) {
        try (NDArray number = manager.create(n)) {
            return pow(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray pow(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.pow(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray xlogy(NDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for xlogy()");
        }
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.xlogy(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray addi(Number n) {
        try (NDArray number = manager.create(n)) {
            return addi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray addi(NDArray other) {
        intern(add(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray subi(Number n) {
        try (NDArray number = manager.create(n)) {
            return subi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray subi(NDArray other) {
        intern(sub(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray muli(Number n) {
        try (NDArray number = manager.create(n)) {
            return muli(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray muli(NDArray other) {
        intern(mul(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray divi(Number n) {
        try (NDArray number = manager.create(n)) {
            return divi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray divi(NDArray other) {
        intern(div(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray modi(Number n) {
        try (NDArray number = manager.create(n)) {
            return modi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray modi(NDArray other) {
        intern(mod(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray powi(Number n) {
        try (NDArray number = manager.create(n)) {
            return powi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray powi(NDArray other) {
        intern(pow(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray signi() {
        intern(sign());
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray negi() {
        intern(neg());
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sign() {
        return toArray(RustLibrary.sign(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray maximum(Number n) {
        try (NDArray number = manager.create(n)) {
            return maximum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray maximum(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.maximum(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray minimum(Number n) {
        try (NDArray number = manager.create(n)) {
            return minimum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray minimum(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.minimum(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray all() {
        NDArray noneZero = countNonzero();
        RsNDArray ret = (RsNDArray) manager.create(noneZero.getLong() == size());
        noneZero.close();
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray any() {
        NDArray noneZero = countNonzero();
        RsNDArray ret = (RsNDArray) manager.create(noneZero.getLong() > 0);
        noneZero.close();
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray none() {
        NDArray noneZero = countNonzero();
        RsNDArray ret = (RsNDArray) manager.create(noneZero.getLong() == 0);
        noneZero.close();
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray countNonzero() {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.countNonzero(getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray countNonzero(int axis) {
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.countNonzeroWithAxis(getHandle(), axis), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray neg() {
        return toArray(RustLibrary.neg(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray abs() {
        return toArray(RustLibrary.abs(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray square() {
        return toArray(RustLibrary.square(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        return toArray(RustLibrary.sqrt(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray cbrt() {
        try (RsNDArray array = (RsNDArray) manager.create(1.0 / 3)) {
            return toArray(RustLibrary.pow(getHandle(), array.getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray floor() {
        return toArray(RustLibrary.floor(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray ceil() {
        return toArray(RustLibrary.ceil(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray round() {
        return toArray(RustLibrary.round(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray trunc() {
        return toArray(RustLibrary.trunc(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray exp() {
        return toArray(RustLibrary.exp(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gammaln() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray log() {
        return toArray(RustLibrary.log(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray log10() {
        return toArray(RustLibrary.log10(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray log2() {
        return toArray(RustLibrary.log2(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sin() {
        return toArray(RustLibrary.sin(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray cos() {
        return toArray(RustLibrary.cos(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray tan() {
        return toArray(RustLibrary.tan(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray asin() {
        return toArray(RustLibrary.asin(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray acos() {
        return toArray(RustLibrary.acos(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray atan() {
        return toArray(RustLibrary.atan(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray atan2(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long otherHandle = manager.from(other).getHandle();
            return toArray(RustLibrary.atan2(getHandle(), otherHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sinh() {
        return toArray(RustLibrary.sinh(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray cosh() {
        return toArray(RustLibrary.cosh(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray tanh() {
        return toArray(RustLibrary.tanh(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray asinh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray acosh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray atanh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray toDegrees() {
        return mul(180.0).div(Math.PI);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray toRadians() {
        return mul(Math.PI).div(180.0);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray max() {
        if (isScalar()) {
            return this;
        }
        return toArray(RustLibrary.max(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray max(int[] axes, boolean keepDims) {
        if (axes.length > 1) {
            // TODO fix this
            throw new UnsupportedOperationException("Only 1 axis is support!");
        }
        return toArray(RustLibrary.maxWithAxis(getHandle(), axes[0], keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray min() {
        if (isScalar()) {
            return this;
        }
        return toArray(RustLibrary.min(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray min(int[] axes, boolean keepDims) {
        if (axes.length > 1) {
            // TODO fix this
            throw new UnsupportedOperationException("Only 1 axis is support!");
        }
        return toArray(RustLibrary.minWithAxis(getHandle(), axes[0], keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sum() {
        if (isScalar()) {
            return this;
        }
        return toArray(RustLibrary.sum(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sum(int[] axes, boolean keepDims) {
        return toArray(RustLibrary.sumWithAxis(getHandle(), axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumProd(int axis) {
        return toArray(RustLibrary.cumProd(getHandle(), axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumProd(int axis, DataType dataType) {
        return toArray(RustLibrary.cumProdWithType(getHandle(), axis, dataType.ordinal()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray prod() {
        return toArray(RustLibrary.prod(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray prod(int[] axes, boolean keepDims) {
        if (axes.length > 1) {
            throw new UnsupportedOperationException("Only 1 axis is support!");
        }
        return toArray(RustLibrary.cumProdWithAxis(getHandle(), axes[0], keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray mean() {
        return toArray(RustLibrary.mean(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray mean(int[] axes, boolean keepDims) {
        return toArray(RustLibrary.meanWithAxis(getHandle(), axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray normalize(double p, long dim, double eps) {
        return toArray(RustLibrary.normalize(getHandle(), p, dim, eps));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray rotate90(int times, int[] axes) {
        if (axes.length != 2) {
            throw new IllegalArgumentException("Axes must be 2");
        }
        return toArray(RustLibrary.rot90(getHandle(), times, axes));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        if (indices.length == 0) {
            return new NDList(this);
        }
        long lastIndex = getShape().get(axis);
        if (indices[indices.length - 1] != lastIndex) {
            long[] tmp = new long[indices.length + 1];
            System.arraycopy(indices, 0, tmp, 0, indices.length);
            tmp[indices.length] = lastIndex;
            indices = tmp;
        }
        return toList(RustLibrary.split(getHandle(), indices, axis));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray flatten() {
        return toArray(RustLibrary.flatten(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten(int startDim, int endDim) {
        return toArray(RustLibrary.flattenWithDims(getHandle(), startDim, endDim));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rfft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ifft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray irfft(long length, long axis) {
        throw new UnsupportedOperationException("Not implemented");
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
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fft2(long[] sizes, long[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pad(Shape padding, double value) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ifft2(long[] sizes, long[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray reshape(Shape shape) {
        long prod = 1;
        int neg = -1;
        long[] dims = shape.getShape();
        for (int i = 0; i < dims.length; ++i) {
            if (dims[i] < 0) {
                if (neg != -1) {
                    throw new IllegalArgumentException("only 1 negative axis is allowed");
                }
                neg = i;
            } else {
                prod *= dims[i];
            }
        }
        if (neg != -1) {
            long total = getShape().size();
            if (total % prod != 0) {
                throw new IllegalArgumentException("unsupported dimensions");
            }
            dims[neg] = total / prod;
        }
        return toArray(RustLibrary.reshape(getHandle(), shape.getShape()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray expandDims(int axis) {
        return toArray(RustLibrary.expandDims(getHandle(), axis));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray squeeze(int[] axes) {
        return toArray(RustLibrary.squeeze(getHandle(), axes));
    }

    /** {@inheritDoc} */
    @Override
    public NDList unique(Integer dim, boolean sorted, boolean returnInverse, boolean returnCounts) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray logicalAnd(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long otherHandle = manager.from(other).getHandle();
            return toArray(RustLibrary.logicalAnd(getHandle(), otherHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray logicalOr(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long otherHandle = manager.from(other).getHandle();
            return toArray(RustLibrary.logicalOr(getHandle(), otherHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray logicalXor(NDArray other) {
        try (NDScope ignore = new NDScope()) {
            long otherHandle = manager.from(other).getHandle();
            return toArray(RustLibrary.logicalXor(getHandle(), otherHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray logicalNot() {
        return toArray(RustLibrary.logicalNot(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray argSort(int axis, boolean ascending) {
        return toArray(RustLibrary.argSort(getHandle(), axis, ascending));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sort() {
        return sort(-1);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sort(int axis) {
        return toArray(RustLibrary.sort(getHandle(), axis, false));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray softmax(int axis) {
        if (getShape().isScalar() || shape.size() == 0) {
            return (RsNDArray) duplicate();
        }
        return toArray(RustLibrary.softmax(getHandle(), axis));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray logSoftmax(int axis) {
        return toArray(RustLibrary.logSoftmax(getHandle(), axis));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray cumSum() {
        // TODO: change default behavior on cumSum
        if (isScalar()) {
            return (RsNDArray) reshape(1);
        }
        if (isEmpty()) {
            return (RsNDArray) reshape(0);
        }
        return cumSum(0);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray cumSum(int axis) {
        if (getShape().dimension() > 3) {
            throw new UnsupportedOperationException("Only 3 dimensions or less is supported");
        }
        return toArray(RustLibrary.cumSum(getHandle(), axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diagonal() {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diagonal(int offset) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diagonal(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        RsNDArray arr = (RsNDArray) replaced;
        Long oldHandle = handle.getAndSet(arr.handle.getAndSet(null));
        RustLibrary.deleteTensor(oldHandle);
        // dereference old ndarray
        arr.close();
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray isInfinite() {
        return toArray(RustLibrary.isInf(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray isNaN() {
        return toArray(RustLibrary.isNaN(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray tile(long repeats) {
        // zero-dim
        if (isEmpty()) {
            return (RsNDArray) duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
        Arrays.fill(repeatsArray, repeats);
        return tile(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray tile(int axis, long repeat) {
        return toArray(RustLibrary.tileWithAxis(getHandle(), axis, repeat));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray tile(long[] repeats) {
        return toArray(RustLibrary.tile(getHandle(), repeats));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray tile(Shape desiredShape) {
        return toArray(RustLibrary.tileWithShape(getHandle(), desiredShape.getShape()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray repeat(long repeats) {
        // zero-dim
        if (isEmpty()) {
            return (RsNDArray) duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
        Arrays.fill(repeatsArray, repeats);
        return repeat(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray repeat(int axis, long repeat) {
        return toArray(RustLibrary.repeat(getHandle(), repeat, axis));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray repeat(long[] repeats) {
        RsNDArray result = this;
        for (int dim = 0; dim < repeats.length; dim++) {
            RsNDArray temp = result;
            result = result.repeat(dim, repeats[dim]);
            if (temp != this) {
                temp.close();
            }
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray repeat(Shape desiredShape) {
        return repeat(repeatsToMatchShape(desiredShape));
    }

    private long[] repeatsToMatchShape(Shape desiredShape) {
        Shape curShape = getShape();
        int dimension = curShape.dimension();
        if (desiredShape.dimension() > dimension) {
            throw new IllegalArgumentException("The desired shape has too many dimensions");
        }
        if (desiredShape.dimension() < dimension) {
            int additionalDimensions = dimension - desiredShape.dimension();
            desiredShape = curShape.slice(0, additionalDimensions).addAll(desiredShape);
        }
        long[] repeats = new long[dimension];
        for (int i = 0; i < dimension; i++) {
            if (curShape.get(i) == 0 || desiredShape.get(i) % curShape.get(i) != 0) {
                throw new IllegalArgumentException(
                        "The desired shape is not a multiple of the original shape");
            }
            repeats[i] = Math.round(Math.ceil((double) desiredShape.get(i) / curShape.get(i)));
        }
        return repeats;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray dot(NDArray other) {
        int selfDim = this.getShape().dimension();
        int otherDim = other.getShape().dimension();
        if (selfDim != otherDim || selfDim > 2) {
            throw new UnsupportedOperationException(
                    "Dimension mismatch or dimension is greater than 2.  Dot product is only"
                            + " applied on two 1D vectors. For high dimensions, please use .matMul"
                            + " instead.");
        }
        try (NDScope ignore = new NDScope()) {
            return toArray(RustLibrary.dot(getHandle(), manager.from(other).getHandle()), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        if (getShape().dimension() < 2 || getShape().dimension() < 2) {
            throw new IllegalArgumentException("only 2d tensors are supported for matMul()");
        }
        try (NDScope ignore = new NDScope()) {
            long otherHandle = manager.from(other).getHandle();
            return toArray(RustLibrary.matmul(getHandle(), otherHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray batchMatMul(NDArray other) {
        if (getShape().dimension() != 3 || getShape().dimension() != 3) {
            throw new IllegalArgumentException("only 3d tensors are allowed for batchMatMul()");
        }
        try (NDScope ignore = new NDScope()) {
            long otherHandle = manager.from(other).getHandle();
            return toArray(RustLibrary.batchMatMul(getHandle(), otherHandle), true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray clip(Number min, Number max) {
        return toArray(RustLibrary.clip(getHandle(), min.doubleValue(), max.doubleValue()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray swapAxes(int axis1, int axis2) {
        return toArray(RustLibrary.transpose(getHandle(), axis1, axis2));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flip(int... axes) {
        return toArray(RustLibrary.flip(getHandle(), axes));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray transpose() {
        int dim = getShape().dimension();
        int[] reversedShape = IntStream.range(0, dim).map(i -> dim - i - 1).toArray();
        return transpose(reversedShape);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray transpose(int... axes) {
        if (isScalar() && axes.length > 0) {
            throw new IllegalArgumentException("axes don't match NDArray");
        }
        return toArray(RustLibrary.permute(getHandle(), axes));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray broadcast(Shape shape) {
        return toArray(RustLibrary.broadcast(getHandle(), shape.getShape()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty NDArray");
        }
        if (isScalar()) {
            return (RsNDArray) manager.create(0L);
        }
        return toArray(RustLibrary.argMax(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray argMax(int axis) {
        if (isScalar()) {
            return (RsNDArray) manager.create(0L);
        }
        return toArray(RustLibrary.argMaxWithAxis(getHandle(), axis, false));
    }

    /** {@inheritDoc} */
    @Override
    public NDList topK(int k, int axis, boolean largest, boolean sorted) {
        return toList(RustLibrary.topK(getHandle(), k, axis, largest, sorted));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty NDArray");
        }
        if (isScalar()) {
            return (RsNDArray) manager.create(0L);
        }
        return toArray(RustLibrary.argMin(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray argMin(int axis) {
        if (isScalar()) {
            return (RsNDArray) manager.create(0L);
        }
        return toArray(RustLibrary.argMinWithAxis(getHandle(), axis, false));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray percentile(Number percentile) {
        return toArray(RustLibrary.percentile(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray percentile(Number percentile, int[] axes) {
        return toArray(RustLibrary.percentileWithAxes(getHandle(), percentile.doubleValue(), axes));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray median() {
        return median(new int[] {-1});
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray median(int[] axes) {
        if (axes.length != 1) {
            throw new UnsupportedOperationException(
                    "Not supporting zero or multi-dimension median");
        }
        NDList result = toList(RustLibrary.median(getHandle(), axes[0], false));
        result.get(1).close();
        return (RsNDArray) result.get(0);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray toDense() {
        return (RsNDArray) duplicate();
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException("Not supported");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray nonzero() {
        return toArray(RustLibrary.nonZero(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray erfinv() {
        return toArray(RustLibrary.erfinv(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray erf() {
        return toArray(RustLibrary.erf(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray inverse() {
        return toArray(RustLibrary.inverse(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(boolean keepDims) {
        return toArray(RustLibrary.norm(getHandle(), 2, new int[] {}, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(int order, int[] axes, boolean keepDims) {
        return toArray(RustLibrary.norm(getHandle(), order, axes, keepDims));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        return toArray(
                RustLibrary.oneHot(getHandle(), depth, onValue, offValue, dataType.ordinal()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray batchDot(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray complex() {
        return toArray(RustLibrary.complex(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray real() {
        return toArray(RustLibrary.real(getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray conj() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArrayEx getNDArrayInternal() {
        if (ndArrayEx == null) {
            throw new UnsupportedOperationException(
                    "NDArray operation is not supported for String tensor");
        }
        return ndArrayEx;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diff(int n, int dim) {
        throw new UnsupportedOperationException("Not implemented yet.");
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
    public void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null && pointer != -1) {
            RustLibrary.deleteTensor(pointer);
        }
        manager.detachInternal(getUid());
        dataRef = null;
    }

    private RsNDArray toArray(long newHandle) {
        return toArray(newHandle, false);
    }

    private RsNDArray toArray(long newHandle, boolean unregister) {
        return toArray(newHandle, null, unregister, false);
    }

    private RsNDArray toArray(
            long newHandle, DataType dataType, boolean unregister, boolean withName) {
        RsNDArray array = new RsNDArray(manager, newHandle, dataType);
        if (withName) {
            array.setName(getName());
        }
        if (unregister) {
            NDScope.unregister(array);
        }
        return array;
    }

    private NDList toList(long[] handles) {
        NDList list = new NDList(handles.length);
        for (long h : handles) {
            list.add(new RsNDArray(manager, h));
        }
        return list;
    }
}
