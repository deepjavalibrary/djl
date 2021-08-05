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
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.UUID;

/** {@code OrtNDArray} is the ONNX Runtime implementation of {@link NDArray}. */
public class OrtNDArray implements NDArrayAdapter {

    private OrtNDManager manager;
    private OnnxTensor tensor;
    private Shape shape;
    private DataType dataType;
    private String name;
    private boolean isClosed;
    private String uid;

    /**
     * Constructs an ONNX Runtime NDArray from a {@link OnnxTensor} (internal. Use {@link NDManager}
     * instead).
     *
     * @param manager the manager to attach the new array to
     * @param tensor the {@link OnnxTensor} to the ONNX Runtime
     */
    OrtNDArray(OrtNDManager manager, OnnxTensor tensor) {
        this.manager = manager;
        this.tensor = tensor;
        uid = UUID.randomUUID().toString();
        manager.attachInternal(uid, this);
    }

    OnnxTensor getTensor() {
        return tensor;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /** {@inheritDoc} */
    @Override
    public String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (dataType == null) {
            dataType = OrtUtils.toDataType(tensor.getInfo().type);
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        // TODO: Support on multiple devices
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = new Shape(tensor.getInfo().getShape());
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (OrtNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (OrtNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = OrtNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public String[] toStringArray() {
        try {
            return (String[]) tensor.getValue();
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return tensor.getByteBuffer().order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isClosed) {
            return "This array is already closed";
        }
        return toDebugString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        tensor.close();
        isClosed = true;
    }
}
