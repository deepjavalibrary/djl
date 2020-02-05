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
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.PairList;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

public class PtNDManager extends BaseNDManager {

    private static final PtNDManager SYSTEM_MANAGER = new SystemManager();

    protected PtNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static PtNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray create(Shape shape, DataType dataType, Device device) {
        return JniUtils.createEmptyNdArray(this, shape, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = data.remaining();
        // int8, uint8, boolean use ByteBuffer, so need to explicitly input DataType
        DataType inputType = DataType.fromBuffer(data);

        int numOfBytes = inputType.getNumOfBytes();
        ByteBuffer buf = allocateDirect(size * numOfBytes);

        switch (inputType) {
            case FLOAT32:
                buf.asFloatBuffer().put((FloatBuffer) data);
                break;
            case FLOAT64:
                buf.asDoubleBuffer().put((DoubleBuffer) data);
                break;
            case UINT8:
            case INT8:
            case BOOLEAN:
                buf.put((ByteBuffer) data);
                break;
            case INT32:
                buf.asIntBuffer().put((IntBuffer) data);
                break;
            case INT64:
                buf.asLongBuffer().put((LongBuffer) data);
                break;
            case FLOAT16:
            default:
                throw new AssertionError("Show never happen");
        }
        return JniUtils.createNdFromByteBuffer(
                this, buf, shape, dataType, SparseFormat.DENSE, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device device) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device device) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device device) {
        return JniUtils.createZerosNdArray(this, shape, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType, Device device) {
        return JniUtils.createOnesNdArray(this, shape, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(
            Number start, Number stop, Number step, DataType dataType, Device device) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(Number start, Number stop, int num, boolean endpoint, Device device) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            Number low, Number high, Shape shape, DataType dataType, Device device) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            Number loc, Number scale, Shape shape, DataType dataType, Device device) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDManager newSubManager(Device dev) {
        PtNDManager manager = new PtNDManager(this, dev);
        attach(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(PtEngine.ENGINE_NAME);
    }

    /** The SystemManager is the root {@link PtNDManager} of which all others are children. */
    private static final class SystemManager extends PtNDManager {

        SystemManager() {
            super(null, Device.defaultDevice());
        }

        /** {@inheritDoc} */
        @Override
        public void attach(String resourceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detach(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
