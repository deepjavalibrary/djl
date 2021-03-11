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
package ai.djl.dlr.engine;

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
import java.nio.FloatBuffer;
import java.util.Arrays;

/** {@code DlrNDManager} is the DLR implementation of {@link NDManager}. */
public class DlrNDManager extends BaseNDManager {

    private static final DlrNDManager SYSTEM_MANAGER = new SystemManager();

    private DlrNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static DlrNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(DlrEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public DlrNDManager newSubManager(Device dev) {
        DlrNDManager manager = new DlrNDManager(this, dev);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public DlrNDArray create(Buffer data, Shape shape, DataType dataType) {
        if (dataType != DataType.FLOAT32 && !(data instanceof FloatBuffer)) {
            throw new UnsupportedOperationException("DLR only supports float32");
        }
        int size = data.remaining();
        int numOfBytes = dataType.getNumOfBytes();
        ByteBuffer bb = ByteBuffer.allocate(size * numOfBytes);
        bb.asFloatBuffer().put((FloatBuffer) data);
        bb.rewind();
        return new DlrNDArray(this, bb, shape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        if (dataType != DataType.FLOAT32) {
            throw new UnsupportedOperationException("DLR only supports float32");
        }
        int size = Math.toIntExact(shape.size());
        float[] data = new float[size];
        return create(data, shape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        if (dataType != DataType.FLOAT32) {
            throw new UnsupportedOperationException("DLR only supports float32");
        }
        int size = Math.toIntExact(shape.size());
        float[] data = new float[size];
        Arrays.fill(data, 1f);

        return create(data, shape);
    }

    /** The SystemManager is the root {@link DlrNDManager} of which all others are children. */
    private static final class SystemManager extends DlrNDManager {

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
