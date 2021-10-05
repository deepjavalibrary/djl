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
package ai.djl.tensorrt.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Float16Utils;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** {@code TrtNDManager} is the TensorRT implementation of {@link NDManager}. */
public class TrtNDManager extends BaseNDManager {

    private static final TrtNDManager SYSTEM_MANAGER = new SystemManager();

    private NDManager alternativeManager;

    private TrtNDManager(NDManager parent, Device device) {
        super(parent, device);
        alternativeManager = getAlternativeManager();
    }

    static TrtNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(TrtEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public TrtNDArray from(NDArray array) {
        if (array == null || array instanceof TrtNDArray) {
            return (TrtNDArray) array;
        }
        return create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public TrtNDManager newSubManager(Device dev) {
        TrtNDManager manager = new TrtNDManager(this, dev);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public TrtNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBufferSize(data, dataType, size);
        if (data.isDirect() && data instanceof ByteBuffer) {
            return new TrtNDArray(this, alternativeManager, (ByteBuffer) data, shape, dataType);
        }

        ByteBuffer bb = allocateDirect(size * dataType.getNumOfBytes());
        BaseNDManager.copyBuffer(data, bb);
        return new TrtNDArray(this, alternativeManager, bb, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        int size = Math.toIntExact(dataType.getNumOfBytes() * shape.size());
        ByteBuffer bb = allocateDirect(size);
        return create(bb, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        int size = (int) shape.size();
        ByteBuffer bb = allocateDirect(size * dataType.getNumOfBytes());
        for (int i = 0; i < size; ++i) {
            switch (dataType) {
                case BOOLEAN:
                case INT8:
                case UINT8:
                    bb.put((byte) 1);
                    break;
                case FLOAT16:
                    bb.putShort(Float16Utils.floatToHalf(1));
                    break;
                case FLOAT32:
                    bb.putFloat(1f);
                    break;
                case FLOAT64:
                    bb.putDouble(1);
                    break;
                case INT32:
                    bb.putInt(1);
                    break;
                case INT64:
                    bb.putLong(1);
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported dataType: " + dataType);
            }
        }
        bb.rewind();
        return create(bb, shape, dataType);
    }

    /** The SystemManager is the root {@link TrtNDManager} of which all others are children. */
    private static final class SystemManager extends TrtNDManager {

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
