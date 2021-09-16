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
package ai.djl.tensorrt.jni;

import ai.djl.Device;
import ai.djl.ndarray.types.DataType;
import java.nio.ByteBuffer;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class containing utilities to interact with the PyTorch Engine's Java Native Interface (JNI)
 * layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    private static final Logger logger = LoggerFactory.getLogger(JniUtils.class);

    private JniUtils() {}

    public static void initPlugins(String namespace) {
        int logLevel = 1;
        if (logger.isWarnEnabled()) {
            logLevel = 2;
        }
        if (logger.isInfoEnabled()) {
            logLevel = 3;
        }
        if (logger.isTraceEnabled()) {
            logLevel = 4;
        }
        TrtLibrary.LIB.initPlugins(namespace, logLevel);
    }

    public static long loadModel(
            int modelType, String path, Device device, Map<String, ?> options) {
        int deviceId = device == null ? 0 : device.getDeviceId();

        String[] keys = new String[options.size()];
        String[] values = new String[keys.length];
        int index = 0;
        String modelTypeName = modelType == 0 ? "ONNX" : "UFF";
        logger.debug("Loading TensorRT {} model {} with options:", modelTypeName, path);
        for (Map.Entry<String, ?> entry : options.entrySet()) {
            keys[index] = entry.getKey();
            values[index] = entry.getValue().toString();
            logger.debug("{}: {}", keys[index], values[index]);
            ++index;
        }
        return TrtLibrary.LIB.loadTrtModel(modelType, path, deviceId, keys, values);
    }

    public static void deleteTrtModel(long model) {
        TrtLibrary.LIB.deleteTrtModel(model);
    }

    public static long createSession(long model) {
        return TrtLibrary.LIB.createSession(model);
    }

    public static void deleteSession(long session) {
        TrtLibrary.LIB.deleteSession(session);
    }

    public static String[] getInputNames(long model) {
        return TrtLibrary.LIB.getInputNames(model);
    }

    public static DataType[] getInputDataTypes(long model) {
        int[] types = TrtLibrary.LIB.getInputDataTypes(model);
        DataType[] ret = new DataType[types.length];
        for (int i = 0; i < types.length; ++i) {
            ret[i] = fromTrt(types[i]);
        }
        return ret;
    }

    public static String[] getOutputNames(long model) {
        return TrtLibrary.LIB.getOutputNames(model);
    }

    public static DataType[] getOutputDataTypes(long model) {
        int[] types = TrtLibrary.LIB.getOutputDataTypes(model);
        DataType[] ret = new DataType[types.length];
        for (int i = 0; i < types.length; ++i) {
            ret[i] = fromTrt(types[i]);
        }
        return ret;
    }

    public static long[] getShape(long session, String name) {
        return TrtLibrary.LIB.getShape(session, name);
    }

    public static void bind(long session, String name, ByteBuffer buffer) {
        TrtLibrary.LIB.bind(session, name, buffer);
    }

    public static void runTrtModel(long session) {
        TrtLibrary.LIB.runTrtModel(session);
    }

    public static String getTrtVersion() {
        int version = TrtLibrary.LIB.getTrtVersion();
        int major = version / 1000;
        int minor = version / 100 - major * 10;
        int patch = version % 100;
        return major + "." + minor + '.' + patch;
    }

    public static DataType fromTrt(int trtType) {
        switch (trtType) {
            case 0:
                return DataType.FLOAT32;
            case 1:
                return DataType.FLOAT16;
            case 2:
                return DataType.INT8;
            case 3:
                return DataType.INT32;
            case 4:
                return DataType.BOOLEAN;
            default:
                throw new UnsupportedOperationException("Unsupported TensorRT type: " + trtType);
        }
    }
}
