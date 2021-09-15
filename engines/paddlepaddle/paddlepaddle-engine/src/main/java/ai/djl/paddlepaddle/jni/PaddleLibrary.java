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
package ai.djl.paddlepaddle.jni;

import java.nio.ByteBuffer;

/** A class containing utilities to interact with the PaddlePaddle Engine's JNI layer. */
@SuppressWarnings("missingjavadocmethod")
final class PaddleLibrary {

    static final PaddleLibrary LIB = new PaddleLibrary();

    private PaddleLibrary() {}

    native long paddleCreateTensor(ByteBuffer data, long length, int[] shape, int dType);

    native void deleteTensor(long handle);

    native int[] getTensorShape(long handle);

    native int getTensorDType(long handle);

    native byte[] getTensorData(long handle);

    native void setTensorName(long handle, String name);

    native String getTensorName(long handle);

    native void setTensorLoD(long handle, long[][] lod);

    native long[][] getTensorLoD(long handle);

    native void loadExtraDir(String[] args);

    native long createAnalysisConfig(String modelDir, String paramDir, int deviceId);

    native void analysisConfigEnableMKLDNN(long handle);

    native void analysisConfigDisableGLog(long handle);

    native void analysisConfigCMLNumThreads(long handle, int threads);

    native void analysisConfigSwitchIrOptim(long handle, boolean condition);

    native void analysisConfigRemovePass(long handle, String pass);

    native void useFeedFetchOp(long handle);

    native void deleteAnalysisConfig(long handle);

    native long createPredictor(long configHandle);

    native long clonePredictor(long handle);

    native void deletePredictor(long handle);

    native String[] getInputNames(long handle);

    native long[] runInference(long handle, long[] inputHandles);
}
