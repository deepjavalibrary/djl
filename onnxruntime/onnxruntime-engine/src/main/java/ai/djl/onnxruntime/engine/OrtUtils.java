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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
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

final class OrtUtils {

    private OrtUtils() {}

    public static OnnxTensor toTensor(OrtEnvironment env, NDArray array) throws OrtException {
        ByteBuffer bb = array.toByteBuffer();
        DataType dataType = array.getDataType();
        Buffer buf = dataType.asDataType(bb);
        return toTensor(env, buf, array.getShape(), dataType);
    }

    public static OnnxTensor toTensor(
            OrtEnvironment env, Buffer data, Shape shape, DataType dataType) throws OrtException {
        if (shape.size() == 0) {
            throw new UnsupportedOperationException("OnnxRuntime doesn't support 0 length tensor.");
        }
        long[] sh = shape.getShape();
        switch (dataType) {
            case FLOAT32:
                return OnnxTensor.createTensor(env, (FloatBuffer) data, sh);
            case FLOAT64:
                return OnnxTensor.createTensor(env, (DoubleBuffer) data, sh);
            case INT32:
                return OnnxTensor.createTensor(env, (IntBuffer) data, sh);
            case INT64:
                return OnnxTensor.createTensor(env, (LongBuffer) data, sh);
            case INT8:
            case UINT8:
                return OnnxTensor.createTensor(env, (ByteBuffer) data, sh, OnnxJavaType.INT8);
            case STRING:
                throw new UnsupportedOperationException(
                        "Use toTensor(OrtEnvironment env, String[] inputs, Shape shape) instead.");
            case BOOLEAN:
            case FLOAT16:
            default:
                throw new UnsupportedOperationException("Data type not supported: " + dataType);
        }
    }

    public static OnnxTensor toTensor(OrtEnvironment env, String[] inputs, Shape shape)
            throws OrtException {
        long[] sh = shape.getShape();
        return OnnxTensor.createTensor(env, inputs, sh);
    }

    public static NDArray toNDArray(NDManager manager, OnnxTensor tensor) {
        if (manager instanceof OrtNDManager) {
            return ((OrtNDManager) manager).create(tensor);
        }
        ByteBuffer bb = tensor.getByteBuffer();
        bb.order(ByteOrder.nativeOrder());
        DataType dataType = OrtUtils.toDataType(tensor.getInfo().type);
        Shape shape = new Shape(tensor.getInfo().getShape());
        Buffer buf = dataType.asDataType(bb);
        tensor.close();
        return manager.create(buf, shape, dataType);
    }

    public static DataType toDataType(OnnxJavaType javaType) {
        switch (javaType) {
            case FLOAT:
                return DataType.FLOAT32;
            case DOUBLE:
                return DataType.FLOAT64;
            case INT8:
                return DataType.INT8;
            case INT32:
                return DataType.INT32;
            case INT64:
                return DataType.INT64;
            case BOOL:
                return DataType.BOOLEAN;
            case UNKNOWN:
                return DataType.UNKNOWN;
            case STRING:
                return DataType.STRING;
            default:
                throw new UnsupportedOperationException("type is not supported: " + javaType);
        }
    }
}
