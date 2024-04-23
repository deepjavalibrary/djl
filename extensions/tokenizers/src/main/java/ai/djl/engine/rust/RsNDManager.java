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
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;

/** {@code PtNDManager} is the Rust implementation of {@link NDManager}. */
public class RsNDManager extends BaseNDManager {

    private static final RsNDManager SYSTEM_MANAGER = new SystemManager();

    private RsNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static RsNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray from(NDArray array) {
        if (array == null || array instanceof RsNDArray) {
            return (RsNDArray) array;
        }
        RsNDArray result = create(array.toByteBuffer(), array.getShape(), array.getDataType());
        result.setName(array.getName());
        return result;
    }

    /**
     * Constructs an RsNDArray from a native handle (internal. Use {@link NDManager} instead).
     *
     * @param data bytebuffer that holds the native memory
     * @param handle the pointer to the native RsNDArray memory
     */
    public RsNDArray createInternal(ByteBuffer data, long handle, DataType dataType) {
        return new RsNDArray(this, handle, dataType, data);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray create(Shape shape, DataType dataType) {
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.zeros(shape.getShape(), dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBuffer(data, dataType, size);
        ByteBuffer buf;
        if (data.isDirect() && data instanceof ByteBuffer) {
            buf = (ByteBuffer) data;
        } else {
            buf = allocateDirect(size * dataType.getNumOfBytes());
            copyBuffer(data, buf);
        }
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.tensorOf(buf, shape.getShape(), dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType, buf);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String[] data, Charset charset, Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCoo(Buffer data, long[][] indices, Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        return create(shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.ones(shape.getShape(), dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.full(value, shape.getShape(), dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(int start, int stop, int step, DataType dataType) {
        return arange((float) start, (float) stop, (float) step, dataType, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        if (Math.signum(stop - start) != Math.signum(step)) {
            return create(new Shape(0), dataType, device);
        }
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.arange(start, stop, step, dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        if (k != 0) {
            throw new UnsupportedOperationException(
                    "index of the diagonal is not supported in Rust");
        }
        if (rows != cols) {
            throw new UnsupportedOperationException("rows must equals to columns in Rust");
        }
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.eye(rows, cols, dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint) {
        if (!endpoint) {
            throw new UnsupportedOperationException("endpoint only support true");
        }
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = DataType.FLOAT32.ordinal();
        long handle = RustLibrary.linspace(start, stop, num, dType, deviceType, deviceId);
        return new RsNDArray(this, handle, DataType.FLOAT32);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomInteger(long low, long high, Shape shape, DataType dataType) {
        long[] sh = shape.getShape();
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = DataType.FLOAT32.ordinal();
        long handle = RustLibrary.randint(low, high, sh, dType, deviceType, deviceId);
        return new RsNDArray(this, handle, DataType.FLOAT32);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomPermutation(long n) {
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        long handle = RustLibrary.randomPermutation(n, deviceType, deviceId);
        return new RsNDArray(this, handle);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        long[] sh = shape.getShape();
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.uniform(low, high, sh, dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        long[] sh = shape.getShape();
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        int dType = toRustDataType(dataType);
        long handle = RustLibrary.randomNormal(loc, scale, sh, dType, deviceType, deviceId);
        return new RsNDArray(this, handle, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray hanningWindow(long numPoints) {
        String deviceType = device.getDeviceType();
        int deviceId = device.getDeviceId();
        long handle = RustLibrary.hannWindow(numPoints, deviceType, deviceId);
        return new RsNDArray(this, handle);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDManager newSubManager(Device device) {
        RsNDManager manager = new RsNDManager(this, device);
        attachUncappedInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(RsEngine.ENGINE_NAME);
    }

    int toRustDataType(DataType dataType) {
        switch (dataType) {
            case BOOLEAN:
            case INT8:
                return DataType.UINT8.ordinal();
            case INT32:
                return DataType.UINT32.ordinal();
            case FLOAT16:
            case BFLOAT16:
            case FLOAT32:
            case FLOAT64:
            case UINT8:
            case UINT32:
            case INT64:
                return dataType.ordinal();
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType);
        }
    }

    /** The SystemManager is the root {@link RsNDManager} of which all others are children. */
    private static final class SystemManager extends RsNDManager implements SystemNDManager {

        SystemManager() {
            super(null, null);
        }
    }
}
