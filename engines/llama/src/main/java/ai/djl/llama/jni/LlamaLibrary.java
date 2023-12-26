/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.llama.jni;

/** Native library for llama.cpp. */
@SuppressWarnings("MissingJavadocMethod")
public final class LlamaLibrary {

    private LlamaLibrary() {}

    public static native long loadModel(String filePath, ModelParameters param);

    public static native void generate(long handle, String prompt, InputParameters param);

    public static native void infill(
            long handle, String prefix, String suffix, InputParameters param);

    public static native Token getNext(long handle, long count, long pos);

    public static native float[] embed(long handle, String prompt);

    public static native int[] encode(long handle, String prompt);

    public static native byte[] decodeBytes(long handle, int[] tokens);

    public static native void delete(long handle);
}
