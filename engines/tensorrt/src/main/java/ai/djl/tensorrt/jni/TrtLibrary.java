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

import java.nio.ByteBuffer;

/** A class containing utilities to interact with the TensorRT Engine's JNI layer. */
@SuppressWarnings("MissingJavadocMethod")
final class TrtLibrary {

    static final TrtLibrary LIB = new TrtLibrary();

    private TrtLibrary() {}

    native void initPlugins(String namespace, int logLevel);

    native long loadTrtModel(
            int modelType,
            String modelPath,
            int deviceId,
            String[] optionKeys,
            String[] optionValues);

    native void deleteTrtModel(long modelHandle);

    native String[] getInputNames(long modelHandle);

    native int[] getInputDataTypes(long modelHandle);

    native String[] getOutputNames(long modelHandle);

    native int[] getOutputDataTypes(long modelHandle);

    native long createSession(long modelHandle);

    native void deleteSession(long sessionHandle);

    native long[] getShape(long sessionHandle, String name);

    native void bind(long sessionHandle, String name, ByteBuffer buffer);

    native void runTrtModel(long sessionHandle);

    native int getTrtVersion();
}
