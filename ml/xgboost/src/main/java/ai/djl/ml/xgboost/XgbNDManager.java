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
package ai.djl.ml.xgboost;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import ml.dmlc.xgboost4j.java.JniUtils;

/** {@code XgbNDManager} is the XGBoost implementation of {@link NDManager}. */
public class XgbNDManager extends BaseNDManager {

    private static final XgbNDManager SYSTEM_MANAGER = new SystemManager();

    private XgbNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static XgbNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager(Device device) {
        XgbNDManager manager = new XgbNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(XgbEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public XgbNDArray create(Buffer data, Shape shape, DataType dataType) {
        if (shape.dimension() != 2) {
            throw new UnsupportedOperationException("Shape must be in two dimension");
        }
        DataType inputType = DataType.fromBuffer(data);
        if (inputType != DataType.FLOAT32) {
            throw new UnsupportedOperationException(
                    "Only Float32 data type supported, actual " + inputType);
        }
        if (data.isDirect() && data instanceof ByteBuffer) {
            // TODO: allow user to set missing value
            long handle = JniUtils.createDMatrix(data, shape, 0.0f);
            return new XgbNDArray(this, handle, shape, SparseFormat.DENSE);
        }

        int size = data.remaining();
        int numOfBytes = inputType.getNumOfBytes();
        ByteBuffer buf = allocateDirect(size * numOfBytes);
        buf.asFloatBuffer().put((FloatBuffer) data);
        buf.rewind();
        long handle = JniUtils.createDMatrix(buf, shape, 0.0f);
        return new XgbNDArray(this, handle, shape, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(
            float[] data, long[] indptr, long[] indices, Shape shape, Device device) {
        if (shape.dimension() != 2) {
            throw new UnsupportedOperationException("Shape must be in two dimension");
        }
        int[] intIndices = Arrays.stream(indices).mapToInt(Math::toIntExact).toArray();
        long handle = JniUtils.createDMatrixCSR(indptr, intIndices, data);
        return new XgbNDArray(this, handle, shape, SparseFormat.CSR);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device device) {
        throw new UnsupportedOperationException(
                "Create from Buffer is not supported, please use create from float[] instead");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        if (dataType != DataType.FLOAT32) {
            throw new UnsupportedOperationException("Only float32 supported");
        }
        if (shape.dimension() != 2) {
            throw new UnsupportedOperationException("Shape must be in two dimension");
        }
        int size = Math.toIntExact(4 * shape.size());
        ByteBuffer buffer = allocateDirect(size);
        return create(dataType.asDataType(buffer), shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        if (dataType != DataType.FLOAT32) {
            throw new UnsupportedOperationException("Only float32 supported");
        }
        if (shape.dimension() != 2) {
            throw new UnsupportedOperationException("Shape must be in two dimension");
        }
        long size = shape.size();
        int bytes = Math.toIntExact(4 * size);
        ByteBuffer buffer = allocateDirect(bytes);
        for (int i = 0; i < size; ++i) {
            buffer.putFloat(1f);
        }
        buffer.rewind();
        return create(dataType.asDataType(buffer), shape, dataType);
    }

    /** The SystemManager is the root {@link XgbNDManager} of which all others are children. */
    private static final class SystemManager extends XgbNDManager {

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
