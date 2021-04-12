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
package ml.dmlc.xgboost4j.java;

import ai.djl.engine.EngineException;
import ai.djl.ml.xgboost.XgbNDArray;
import ai.djl.ml.xgboost.XgbNDManager;
import ai.djl.ml.xgboost.XgbSymbolBlock;
import ai.djl.ndarray.types.Shape;
import com.sun.jna.Native;
import com.sun.jna.PointerProxy;
import java.nio.Buffer;

/** DJL class that has access to XGBoost JNI. */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    private JniUtils() {}

    public static void checkCall(int ret) {
        try {
            XGBoostJNI.checkCall(ret);
        } catch (XGBoostError e) {
            throw new EngineException("XGBoost Engine error: ", e);
        }
    }

    public static XgbSymbolBlock loadModel(XgbNDManager manager, String modelPath) {
        // TODO: add Matrix handle option
        long handle = createBoosterHandle(null);
        checkCall(XGBoostJNI.XGBoosterLoadModel(handle, modelPath));

        return new XgbSymbolBlock(manager, handle);
    }

    public static long createDMatrix(Buffer buf, Shape shape, float missing) {
        long[] handles = new long[1];
        int rol = (int) shape.get(0);
        int col = (int) shape.get(1);
        long handle = new PointerProxy(Native.getDirectBufferPointer(buf)).getPeer();
        checkCall(XGBoostJNI.XGDMatrixCreateFromMatRef(handle, rol, col, missing, handles));
        return handles[0];
    }

    public static long createDMatrixCSR(long[] indptr, int[] indices, float[] array) {
        long[] handles = new long[1];
        checkCall(XGBoostJNI.XGDMatrixCreateFromCSREx(indptr, indices, array, 0, handles));
        return handles[0];
    }

    public static void deleteDMatrix(long handle) {
        checkCall(XGBoostJNI.XGDMatrixFree(handle));
    }

    public static float[] inference(
            XgbSymbolBlock block, XgbNDArray array, int treeLimit, XgbSymbolBlock.Mode mode) {
        float[][] output = new float[1][];
        checkCall(
                XGBoostJNI.XGBoosterPredict(
                        block.getHandle(), array.getHandle(), treeLimit, mode.getValue(), output));
        return output[0];
    }

    public static void deleteModel(long handle) {
        checkCall(XGBoostJNI.XGBoosterFree(handle));
    }

    private static long createBoosterHandle(long[] matrixHandles) {
        long[] handles = new long[1];
        checkCall(XGBoostJNI.XGBoosterCreate(matrixHandles, handles));
        return handles[0];
    }
}
