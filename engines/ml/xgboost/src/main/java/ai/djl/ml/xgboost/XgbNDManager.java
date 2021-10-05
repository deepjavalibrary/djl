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

    private NDManager alternativeManager;

    private XgbNDManager(NDManager parent, Device device) {
        super(parent, device);
        alternativeManager = getAlternativeManager();
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
    public XgbNDArray from(NDArray array) {
        if (array == null || array instanceof XgbNDArray) {
            return (XgbNDArray) array;
        }
        return (XgbNDArray) create(array.toByteBuffer(), array.getShape(), array.getDataType());
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
    public NDArray create(Buffer data, Shape shape, DataType dataType) {
        if (shape.dimension() != 2) {
            if (data instanceof ByteBuffer) {
                // output only NDArray
                return new XgbNDArray(this, alternativeManager, (ByteBuffer) data, shape, dataType);
            }
            if (alternativeManager != null) {
                return alternativeManager.create(data, shape, dataType);
            }
            throw new UnsupportedOperationException("XgbNDArray shape must be in two dimension.");
        }
        if (dataType != DataType.FLOAT32) {
            if (data instanceof ByteBuffer) {
                // output only NDArray
                return new XgbNDArray(this, alternativeManager, (ByteBuffer) data, shape, dataType);
            }
            if (alternativeManager != null) {
                return alternativeManager.create(data, shape, dataType);
            }
            throw new UnsupportedOperationException("XgbNDArray only supports float32.");
        }

        if (data.isDirect() && data instanceof ByteBuffer) {
            // TODO: allow user to set missing value
            long handle = JniUtils.createDMatrix(data, shape, 0.0f);
            return new XgbNDArray(this, alternativeManager, handle, shape, SparseFormat.DENSE);
        }

        DataType inputType = DataType.fromBuffer(data);
        if (inputType != DataType.FLOAT32) {
            throw new UnsupportedOperationException(
                    "Only Float32 data type supported, actual " + inputType);
        }

        int size = Math.toIntExact(shape.size() * DataType.FLOAT32.getNumOfBytes());
        ByteBuffer buf = allocateDirect(size);
        buf.asFloatBuffer().put((FloatBuffer) data);
        buf.rewind();
        long handle = JniUtils.createDMatrix(buf, shape, 0.0f);
        return new XgbNDArray(this, alternativeManager, handle, shape, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(Buffer buffer, long[] indptr, long[] indices, Shape shape) {
        if (shape.dimension() != 2) {
            throw new UnsupportedOperationException("Shape must be in two dimension");
        }
        int[] intIndices = Arrays.stream(indices).mapToInt(Math::toIntExact).toArray();
        float[] data = new float[buffer.remaining()];
        ((FloatBuffer) buffer).get(data);
        long handle = JniUtils.createDMatrixCSR(indptr, intIndices, data);
        return new XgbNDArray(this, alternativeManager, handle, shape, SparseFormat.CSR);
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
