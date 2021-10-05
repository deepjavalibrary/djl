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
package ai.djl.paddlepaddle.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.jni.JniUtils;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** {@code PpNDManager} is the PaddlePaddle implementation of {@link NDManager}. */
public class PpNDManager extends BaseNDManager {

    private static final PpNDManager SYSTEM_MANAGER = new SystemManager();

    private NDManager alternativeManager;

    private PpNDManager(NDManager parent, Device device) {
        super(parent, device);
        alternativeManager = getAlternativeManager();
    }

    static PpNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public PpNDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public PpNDManager newSubManager(Device device) {
        PpNDManager manager = new PpNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(PpEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public PpNDArray from(NDArray array) {
        if (array == null || array instanceof PpNDArray) {
            return (PpNDArray) array;
        }
        return create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    /**
     * Creates a new instance of {@code PpNDArray}.
     *
     * <p>For internal use only.
     *
     * @param data bytebuffer that holds the native memory
     * @param handle the pointer to the native MxNDArray memory
     * @return a new instance of {@code PpNDArray}
     */
    public PpNDArray createInternal(ByteBuffer data, long handle) {
        return new PpNDArray(this, alternativeManager, data, handle);
    }

    /** {@inheritDoc} */
    @Override
    public PpNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBufferSize(data, dataType, size);
        if (data.isDirect() && data instanceof ByteBuffer) {
            return JniUtils.createNdArray(this, (ByteBuffer) data, shape, dataType);
        }
        ByteBuffer buf = allocateDirect(size * dataType.getNumOfBytes());
        copyBuffer(data, buf);
        return JniUtils.createNdArray(this, buf, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        if (alternativeManager != null) {
            alternativeManager.close();
            alternativeManager = null;
        }
    }

    /** The SystemManager is the root {@link PpNDManager} of which all others are children. */
    private static final class SystemManager extends PpNDManager {

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
