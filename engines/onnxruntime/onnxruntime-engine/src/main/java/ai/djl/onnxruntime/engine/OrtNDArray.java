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
import java.nio.charset.Charset;
import java.util.UUID;

/** {@code OrtNDArray} is the ONNX Runtime implementation of {@link NDArray}. */
public class OrtNDArray extends NDArrayAdapter {

    private OnnxTensor tensor;

    /**
     * Constructs an ONNX Runtime NDArray from a {@link OnnxTensor} (internal. Use {@link NDManager}
     * instead).
     *
     * @param manager the manager to attach the new array to
     * @param alternativeManager the alternative manager to execute unsupported operation
     * @param tensor the {@link OnnxTensor} to the ONNX Runtime
     */
    OrtNDArray(OrtNDManager manager, NDManager alternativeManager, OnnxTensor tensor) {
        super(manager, alternativeManager, null, null, UUID.randomUUID().toString());
        this.tensor = tensor;
        manager.attachInternal(uid, this);
    }

    OnnxTensor getTensor() {
        return tensor;
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
    public Shape getShape() {
        if (shape == null) {
            shape = new Shape(tensor.getInfo().getShape());
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        tensor.close();
        tensor = ((OrtNDArray) replaced).tensor;
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = OrtNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public String[] toStringArray(Charset charset) {
        try {
            Object obj = tensor.getValue();
            if (obj instanceof String) {
                // Scalar type;
                return new String[] {(String) obj};
            }
            return (String[]) obj;
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        if (getDataType() == DataType.STRING) {
            throw new IllegalArgumentException("Please use toStringArray() for String NDArray.");
        }
        return tensor.getByteBuffer().order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!isClosed) {
            tensor.close();
            super.close();
        }
    }
}
