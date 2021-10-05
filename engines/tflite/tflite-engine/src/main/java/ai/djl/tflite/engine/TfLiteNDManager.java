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
package ai.djl.tflite.engine;

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
import org.tensorflow.lite.Tensor;

/** {@code TfLiteNDManager} is the TFLite implementation of {@link NDManager}. */
public class TfLiteNDManager extends BaseNDManager {

    private static final TfLiteNDManager SYSTEM_MANAGER = new SystemManager();

    private NDManager alternativeManager;

    private TfLiteNDManager(NDManager parent, Device device) {
        super(parent, device);
        alternativeManager = getAlternativeManager();
    }

    static TfLiteNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public TfLiteNDArray from(NDArray array) {
        if (array == null || array instanceof TfLiteNDArray) {
            return (TfLiteNDArray) array;
        }
        return create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    TfLiteNDArray createInternal(Tensor tensor) {
        return new TfLiteNDArray(this, alternativeManager, tensor);
    }

    /** {@inheritDoc} */
    @Override
    public TfLiteNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBufferSize(data, dataType, size);
        if (data.isDirect() && data instanceof ByteBuffer) {
            return new TfLiteNDArray(this, alternativeManager, (ByteBuffer) data, shape, dataType);
        }

        ByteBuffer buf = allocateDirect(size * dataType.getNumOfBytes());
        copyBuffer(data, buf);
        return new TfLiteNDArray(this, alternativeManager, buf, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public TfLiteNDManager newSubManager(Device device) {
        TfLiteNDManager manager = new TfLiteNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(TfLiteEngine.ENGINE_NAME);
    }

    /** The SystemManager is the root {@link TfLiteNDManager} of which all others are children. */
    private static final class SystemManager extends TfLiteNDManager {

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
