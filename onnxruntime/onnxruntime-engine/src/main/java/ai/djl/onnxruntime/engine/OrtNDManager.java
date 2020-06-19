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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;

/** {@code OrtNDManager} is the ONNX Runtime implementation of {@link NDManager}. */
public class OrtNDManager extends BaseNDManager {

    private static final OrtNDManager SYSTEM_MANAGER = new SystemManager();
    OrtEnvironment env;

    private OrtNDManager(NDManager parent, Device device, OrtEnvironment env) {
        super(parent, device);
        this.env = env;
    }

    static OrtNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    OrtNDArray create(OnnxTensor tensor) {
        return new OrtNDArray(this, tensor);
    }

    /** {@inheritDoc} */
    @Override
    public OrtNDArray create(Buffer data, Shape shape, DataType dataType) {
        try {
            switch (dataType) {
                case FLOAT32:
                    return create(
                            OnnxTensor.createTensor(env, (FloatBuffer) data, shape.getShape()));
                case FLOAT64:
                    return create(
                            OnnxTensor.createTensor(env, (DoubleBuffer) data, shape.getShape()));
                case INT32:
                    return create(OnnxTensor.createTensor(env, (IntBuffer) data, shape.getShape()));
                case INT64:
                    return create(
                            OnnxTensor.createTensor(env, (LongBuffer) data, shape.getShape()));
                case INT8:
                case UINT8:
                    return create(
                            OnnxTensor.createTensor(
                                    env, (ByteBuffer) data, shape.getShape(), OnnxJavaType.INT8));
                case BOOLEAN:
                    return create(
                            OnnxTensor.createTensor(
                                    env, (ByteBuffer) data, shape.getShape(), OnnxJavaType.BOOL));
                case FLOAT16:
                default:
                    throw new AssertionError("Data type not supported!");
            }
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public OrtNDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public OrtNDManager newSubManager(Device device) {
        OrtNDManager manager = new OrtNDManager(this, device, env);
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
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** The SystemManager is the root {@link OrtNDManager} of which all others are children. */
    private static final class SystemManager extends OrtNDManager {

        SystemManager() {
            super(null, Device.defaultDevice(), OrtEnvironment.getEnvironment());
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
