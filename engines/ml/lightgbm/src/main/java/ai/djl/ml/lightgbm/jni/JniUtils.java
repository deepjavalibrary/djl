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
package ai.djl.ml.lightgbm.jni;

import ai.djl.engine.EngineException;
import ai.djl.ml.lightgbm.LgbmDataset;
import ai.djl.ml.lightgbm.LgbmNDArray;
import ai.djl.ml.lightgbm.LgbmNDManager;
import ai.djl.ml.lightgbm.LgbmSymbolBlock;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Pair;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_double;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_int;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_long_long;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import com.microsoft.ml.lightgbm.lightgbmlibJNI;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

/** DJL class that has access to LightGBM JNI. */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    private JniUtils() {}

    public static void checkCall(int result) {
        if (result != 0) {
            throw new EngineException("LightGBM Engine Error: " + lightgbmlib.LGBM_GetLastError());
        }
    }

    public static LgbmSymbolBlock loadModel(LgbmNDManager manager, String path) {
        SWIGTYPE_p_p_void handle = lightgbmlib.new_voidpp();
        SWIGTYPE_p_int outIterations = lightgbmlib.new_intp();
        int result = lightgbmlib.LGBM_BoosterCreateFromModelfile(path, outIterations, handle);
        checkCall(result);
        int iterations = lightgbmlib.intp_value(outIterations);
        lightgbmlib.delete_intp(outIterations);
        return new LgbmSymbolBlock(manager, iterations, handle);
    }

    public static void freeModel(SWIGTYPE_p_p_void handle) {
        int result = lightgbmlib.LGBM_BoosterFree(lightgbmlib.voidpp_value(handle));
        checkCall(result);
    }

    public static Pair<Integer, ByteBuffer> inference(
            SWIGTYPE_p_p_void model, int iterations, NDArray a) {
        if (a instanceof LgbmDataset) {
            LgbmDataset dataset = (LgbmDataset) a;
            switch (dataset.getSrcType()) {
                case FILE:
                    throw new IllegalArgumentException(
                            "LightGBM can only do inference with an Array LightGBMDataset");
                case ARRAY:
                    return inferenceMat(model, iterations, dataset.getSrcArrayConverted());
                default:
                    throw new IllegalArgumentException("Unexpected LgbmDataset SrcType");
            }
        }
        if (a instanceof LgbmNDArray) {
            return inferenceMat(model, iterations, (LgbmNDArray) a);
        }
        throw new IllegalArgumentException("LightGBM inference must be called with a LgbmNDArray");
    }

    public static Pair<Integer, ByteBuffer> inferenceMat(
            SWIGTYPE_p_p_void model, int iterations, LgbmNDArray a) {
        SWIGTYPE_p_long_long outLength = lightgbmlib.new_int64_tp();
        SWIGTYPE_p_double outBuffer = null;
        try {
            outBuffer = lightgbmlib.new_doubleArray(2L * a.getRows());
            int result =
                    lightgbmlib.LGBM_BoosterPredictForMat(
                            lightgbmlib.voidpp_value(model),
                            a.getHandle(),
                            a.getTypeConstant(),
                            a.getRows(),
                            a.getCols(),
                            1,
                            lightgbmlibJNI.C_API_PREDICT_NORMAL_get(),
                            0,
                            iterations,
                            "",
                            outLength,
                            outBuffer);
            checkCall(result);
            int length = Math.toIntExact(lightgbmlib.int64_tp_value(outLength));
            if (a.getDataType() == DataType.FLOAT32) {
                ByteBuffer bb = ByteBuffer.allocateDirect(length * 4);
                FloatBuffer wrapped = bb.asFloatBuffer();
                for (int i = 0; i < length; i++) {
                    wrapped.put((float) lightgbmlib.doubleArray_getitem(outBuffer, i));
                }
                bb.rewind();
                return new Pair<>(length, bb);
            } else if (a.getDataType() == DataType.FLOAT64) {
                ByteBuffer bb = ByteBuffer.allocateDirect(length * 8);
                DoubleBuffer wrapped = bb.asDoubleBuffer();
                for (int i = 0; i < length; i++) {
                    wrapped.put(lightgbmlib.doubleArray_getitem(outBuffer, i));
                }
                bb.rewind();
                return new Pair<>(length, bb);
            } else {
                throw new IllegalArgumentException(
                        "Unexpected data type for LightGBM inference. Expected Float32 or Float64,"
                                + " but found "
                                + a.getDataType());
            }
        } catch (EngineException e) {
            throw new EngineException("Failed to run inference using LightGBM native engine", e);
        } finally {
            lightgbmlib.delete_int64_tp(outLength);
            if (outBuffer != null) {
                lightgbmlib.delete_doubleArray(outBuffer);
            }
        }
    }

    public static SWIGTYPE_p_p_void datasetFromFile(String fileName) {
        SWIGTYPE_p_p_void handle = lightgbmlib.new_voidpp();
        int result = lightgbmlib.LGBM_DatasetCreateFromFile(fileName, "", null, handle);
        checkCall(result);
        return handle;
    }

    public static SWIGTYPE_p_p_void datasetFromArray(LgbmNDArray a) {
        SWIGTYPE_p_p_void handle = lightgbmlib.new_voidpp();
        int result =
                lightgbmlib.LGBM_DatasetCreateFromMat(
                        a.getHandle(),
                        a.getTypeConstant(),
                        a.getRows(),
                        a.getCols(),
                        1,
                        "",
                        null,
                        handle);
        checkCall(result);
        return handle;
    }

    public static int datasetGetRows(SWIGTYPE_p_p_void handle) {
        SWIGTYPE_p_int outp = lightgbmlib.new_intp();
        try {
            int result = lightgbmlib.LGBM_DatasetGetNumData(lightgbmlib.voidpp_value(handle), outp);
            checkCall(result);
            return lightgbmlib.intp_value(outp);
        } finally {
            lightgbmlib.delete_intp(outp);
        }
    }

    public static int datasetGetCols(SWIGTYPE_p_p_void handle) {
        SWIGTYPE_p_int outp = lightgbmlib.new_intp();
        try {
            int result =
                    lightgbmlib.LGBM_DatasetGetNumFeature(lightgbmlib.voidpp_value(handle), outp);
            checkCall(result);
            return lightgbmlib.intp_value(outp);
        } finally {
            lightgbmlib.delete_intp(outp);
        }
    }

    public static void freeDataset(SWIGTYPE_p_p_void handle) {
        int result = lightgbmlib.LGBM_DatasetFree(lightgbmlib.voidpp_value(handle));
        checkCall(result);
    }
}
