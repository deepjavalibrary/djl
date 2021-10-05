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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

final class OrtUtils {

    private OrtUtils() {}

    public static OnnxTensor toTensor(
            OrtEnvironment env, Buffer data, Shape shape, DataType dataType) {
        if (shape.size() == 0) {
            throw new UnsupportedOperationException("OnnxRuntime doesn't support 0 length tensor.");
        }
        if (data instanceof ByteBuffer) {
            data = dataType.asDataType((ByteBuffer) data);
        }
        long[] sh = shape.getShape();
        try {
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
                    return OnnxTensor.createTensor(env, (ByteBuffer) data, sh, OnnxJavaType.INT8);
                case UINT8:
                    return OnnxTensor.createTensor(env, (ByteBuffer) data, sh, OnnxJavaType.UINT8);
                case STRING:
                    throw new UnsupportedOperationException(
                            "Use toTensor(OrtEnvironment env, String[] inputs, Shape shape) instead.");
                case BOOLEAN:
                case FLOAT16:
                default:
                    throw new UnsupportedOperationException("Data type not supported: " + dataType);
            }
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    public static OnnxTensor toTensor(OrtEnvironment env, String[] inputs, Shape shape)
            throws OrtException {
        long[] sh = shape.getShape();
        return OnnxTensor.createTensor(env, inputs, sh);
    }

    public static DataType toDataType(OnnxJavaType javaType) {
        switch (javaType) {
            case FLOAT:
                return DataType.FLOAT32;
            case DOUBLE:
                return DataType.FLOAT64;
            case INT8:
                return DataType.INT8;
            case UINT8:
                return DataType.UINT8;
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
