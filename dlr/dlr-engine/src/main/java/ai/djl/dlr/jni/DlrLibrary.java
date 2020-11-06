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
package ai.djl.dlr.jni;

/** A class containing utilities to interact with the DLR Engine's JNI layer. */
@SuppressWarnings("MissingJavadocMethod")
final class DlrLibrary {

    static final DlrLibrary LIB = new DlrLibrary();

    private DlrLibrary() {}

    native int getDlrNumInputs(long handle);

    native int getDlrNumWeights(long handle);

    native String getDlrInputName(long handle, int index);

    native String getDlrWeightName(long handle, int index);

    native int getDlrInput(long handle, String jname, long[] shape, float[] input, int dim);

    native int getDlrInput(long handle, String jname, float[] input);

    native int getDlrOutputShape(long jhandle, int index, long[] shape);

    native int getDlrOutput(long handle, int index, float[] output);

    native int getDlrOutputDim(long handle, int index);

    native long getDlrOutputSize(long handle, int index);

    native int getDlrNumOutputs(long handle);

    native long createDlrModel(String modelPath, int deviceType, int deviceId);

    native int deleteDlrModel(long handle);

    native int runDlrModel(long handle);

    native String dlrGetLastError();

    native String getDlrBackend(long handle);

    native int setDlrNumThreads(long handle, int threads);

    native int useDlrCPUAffinity(long handle, boolean use);
}
