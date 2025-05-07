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
        long[] sh = shape.getShape();
        try {
            switch (dataType) {
                case FLOAT32:
                    return OnnxTensor.createTensor(env, asFloatBuffer(data), sh);
                case FLOAT64:
                    return OnnxTensor.createTensor(env, asDoubleBuffer(data), sh);
                case FLOAT16:
                    return OnnxTensor.createTensor(
                            env, (ByteBuffer) data, sh, OnnxJavaType.FLOAT16);
                case BFLOAT16:
                    return OnnxTensor.createTensor(
                            env, (ByteBuffer) data, sh, OnnxJavaType.BFLOAT16);
                case INT32:
                    return OnnxTensor.createTensor(env, asIntBuffer(data), sh);
                case INT64:
                    return OnnxTensor.createTensor(env, asLongBuffer(data), sh);
                case INT8:
                    return OnnxTensor.createTensor(env, (ByteBuffer) data, sh, OnnxJavaType.INT8);
                case UINT8:
                    return OnnxTensor.createTensor(env, (ByteBuffer) data, sh, OnnxJavaType.UINT8);
                case BOOLEAN:
                    return OnnxTensor.createTensor(env, (ByteBuffer) data, sh, OnnxJavaType.BOOL);
                case STRING:
                    throw new UnsupportedOperationException(
                            "Use toTensor(OrtEnvironment env, String[] inputs, Shape shape)"
                                    + " instead.");
                default:
                    throw new UnsupportedOperationException("Data type not supported: " + dataType);
            }
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    public static OnnxTensor toTensor(OrtEnvironment env, String[] inputs, Shape shape)
            throws OrtException {
        return OnnxTensor.createTensor(env, inputs, shape.getShape());
    }

    public static OnnxTensor toTensor(OrtEnvironment env, Object inputs) throws OrtException {
        return OnnxTensor.createTensor(env, inputs);
    }

    public static DataType toDataType(OnnxJavaType javaType) {
        switch (javaType) {
            case FLOAT:
                return DataType.FLOAT32;
            case FLOAT16:
                return DataType.FLOAT16;
            case BFLOAT16:
                return DataType.BFLOAT16;
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

    private static FloatBuffer asFloatBuffer(Buffer data) {
        if (data instanceof ByteBuffer) {
            return ((ByteBuffer) data).asFloatBuffer();
        }
        return (FloatBuffer) data;
    }

    private static DoubleBuffer asDoubleBuffer(Buffer data) {
        if (data instanceof ByteBuffer) {
            return ((ByteBuffer) data).asDoubleBuffer();
        }
        return (DoubleBuffer) data;
    }

    private static IntBuffer asIntBuffer(Buffer data) {
        if (data instanceof ByteBuffer) {
            return ((ByteBuffer) data).asIntBuffer();
        }
        return (IntBuffer) data;
    }

    private static LongBuffer asLongBuffer(Buffer data) {
        if (data instanceof ByteBuffer) {
            return ((ByteBuffer) data).asLongBuffer();
        }
        return (LongBuffer) data;
    }
}
