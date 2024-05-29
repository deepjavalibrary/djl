/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ml.lightgbm;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Path;

/** {@code LgbmNDManager} is the LightGBM implementation of {@link NDManager}. */
public class LgbmNDManager extends BaseNDManager {

    private static final LgbmNDManager SYSTEM_MANAGER = new SystemManager();

    private LgbmNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static LgbmNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public LgbmNDArray from(NDArray array) {
        if (array == null || array instanceof LgbmNDArray) {
            return (LgbmNDArray) array;
        }
        LgbmNDArray result =
                (LgbmNDArray) create(array.toByteBuffer(), array.getShape(), array.getDataType());
        result.setName(array.getName());
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager(Device device) {
        LgbmNDManager manager = new LgbmNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(LgbmEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Buffer data, Shape shape, DataType dataType) {
        if (data instanceof ByteBuffer) {
            // output only NDArray
            return new LgbmNDArray(this, alternativeManager, (ByteBuffer) data, shape, dataType);
        } else if (data instanceof FloatBuffer && dataType == DataType.FLOAT32) {
            ByteBuffer bb = allocateDirect(data.capacity() * 4);
            bb.asFloatBuffer().put((FloatBuffer) data);
            bb.rewind();
            return new LgbmNDArray(this, alternativeManager, bb, shape, dataType);
        } else if (data instanceof DoubleBuffer && dataType == DataType.FLOAT64) {
            ByteBuffer bb = allocateDirect(data.capacity() * 8);
            bb.asDoubleBuffer().put((DoubleBuffer) data);
            bb.rewind();
            return new LgbmNDArray(this, alternativeManager, bb, shape, dataType);
        }
        if (alternativeManager != null) {
            return alternativeManager.create(data, shape, dataType);
        }
        throw new UnsupportedOperationException(
                "LgbmNDArray only supports float32 and float64. Please pass either a ByteBuffer, a"
                        + " FloatBuffer with Float32, or a DoubleBuffer with Float64.");
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        return new NDList(new LgbmDataset(this, null, path));
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

    /** The SystemManager is the root {@link LgbmNDManager} of which all others are children. */
    private static final class SystemManager extends LgbmNDManager implements SystemNDManager {

        SystemManager() {
            super(null, null);
        }
    }
}
