/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.pytorch.jni.JniUtils;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** {@code PtNDManager} is the PyTorch implementation of {@link NDManager}. */
public class PtNDManager extends BaseNDManager {

    private static final PtNDManager SYSTEM_MANAGER = new SystemManager();

    private PtNDManager(NDManager parent, Device device) {
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
    public PtNDArray from(NDArray array) {
        if (array == null || array instanceof PtNDArray) {
            return (PtNDArray) array;
        }
        return create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray create(Shape shape, DataType dataType) {
        return JniUtils.createEmptyNdArray(this, shape, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBufferSize(data, dataType, size);
        if (data.isDirect() && data instanceof ByteBuffer) {
            return JniUtils.createNdFromByteBuffer(
                    this, (ByteBuffer) data, shape, dataType, SparseFormat.DENSE, device);
        }
        ByteBuffer buf = allocateDirect(size * dataType.getNumOfBytes());
        copyBuffer(data, buf);
        return JniUtils.createNdFromByteBuffer(
                this, buf, shape, dataType, SparseFormat.DENSE, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCoo(Buffer data, long[][] indices, Shape shape) {
        // length should be the same as indices dim 1
        try (NDArray valueNd = create(data, new Shape(indices[0].length))) {
            try (NDArray indicesNd = create(indices)) {
                return JniUtils.createSparseCoo((PtNDArray) indicesNd, (PtNDArray) valueNd, shape);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        return JniUtils.createZerosNdArray(this, shape, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        return JniUtils.createOnesNdArray(this, shape, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        return JniUtils.full(this, shape, value, dataType, device, SparseFormat.DENSE);
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
        return JniUtils.arange(this, start, stop, step, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        if (k != 0) {
            throw new UnsupportedOperationException(
                    "index of the diagonal is not supported in PyTorch");
        }
        return JniUtils.eye(this, rows, cols, dataType, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint) {
        if (!endpoint) {
            throw new UnsupportedOperationException("endpoint only support true");
        }
        return JniUtils.linspace(
                this, start, stop, num, DataType.FLOAT32, device, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomInteger(long low, long high, Shape shape, DataType dataType) {
        return JniUtils.randint(this, low, high, shape, dataType, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        return JniUtils.uniform(this, low, high, shape, dataType, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        return JniUtils.normal(this, loc, scale, shape, dataType, device);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDManager newSubManager(Device device) {
        PtNDManager manager = new PtNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(PtEngine.ENGINE_NAME);
    }

    /** The SystemManager is the root {@link PtNDManager} of which all others are children. */
    private static final class SystemManager extends PtNDManager {

        SystemManager() {
            super(null, null);
        }

        /** {@inheritDoc} */
        @Override
        public void attachInternal(String resourceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detachInternal(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
