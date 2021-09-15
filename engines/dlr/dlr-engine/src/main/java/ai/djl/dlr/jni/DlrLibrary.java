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

    native void setDLRInput(long handle, String name, long[] shape, float[] input, int dim);

    native long[] getDlrOutputShape(long handle, int index);

    native float[] getDlrOutput(long handle, int index);

    native int getDlrNumOutputs(long handle);

    native long createDlrModel(String modelPath, int deviceType, int deviceId);

    native void deleteDlrModel(long handle);

    native void runDlrModel(long handle);

    native String getDlrBackend(long handle);

    native String getDlrVersion();

    native void setDlrNumThreads(long handle, int threads);

    native void useDlrCPUAffinity(long handle, boolean use);
}
