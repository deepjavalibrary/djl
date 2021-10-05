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
package ai.djl.onnxruntime.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;

/** {@code OrtNDManager} is the ONNX Runtime implementation of {@link NDManager}. */
public class OrtNDManager extends BaseNDManager {

    private static final OrtNDManager SYSTEM_MANAGER = new SystemManager();

    private OrtEnvironment env;
    private NDManager alternativeManager;

    private OrtNDManager(NDManager parent, Device device, OrtEnvironment env) {
        super(parent, device);
        this.env = env;
        alternativeManager = getAlternativeManager();
    }

    static OrtNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public OrtNDArray from(NDArray array) {
        if (array == null || array instanceof OrtNDArray) {
            return (OrtNDArray) array;
        }
        return create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    OrtNDArray createInternal(OnnxTensor tensor) {
        return new OrtNDArray(this, alternativeManager, tensor);
    }

    /** {@inheritDoc} */
    @Override
    public OrtNDArray create(Buffer data, Shape shape, DataType dataType) {
        if (dataType == DataType.STRING) {
            throw new IllegalArgumentException(
                    "Use NDManager.create(String[], Shape) to create String NDArray.");
        }
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBufferSize(data, dataType, size);
        OnnxTensor tensor = OrtUtils.toTensor(env, data, shape, dataType);
        return new OrtNDArray(this, alternativeManager, tensor);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String data) {
        return create(new String[] {data});
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String[] data) {
        return create(data, new Shape(data.length));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String[] data, Charset charset, Shape shape) {
        try {
            return new OrtNDArray(this, alternativeManager, OrtUtils.toTensor(env, data, shape));
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public OrtNDManager newSubManager(Device device) {
        OrtNDManager manager = new OrtNDManager(this, device, env);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(OrtEngine.ENGINE_NAME);
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

    /** The SystemManager is the root {@link OrtNDManager} of which all others are children. */
    private static final class SystemManager extends OrtNDManager {

        SystemManager() {
            super(null, null, OrtEnvironment.getEnvironment());
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
