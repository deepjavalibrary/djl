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
package ai.djl.fasttext.jni;

/** A class containing utilities to interact with the SentencePiece Engine's JNI layer. */
@SuppressWarnings("MissingJavadocMethod")
final class FastTextLibrary {

    static final FastTextLibrary LIB = new FastTextLibrary();

    private FastTextLibrary() {}

    native Pointer createFastText();

    native void freeFastText(Pointer handle);

    native void loadModel(Pointer handle, String filePath);

    native boolean checkModel(String filePath);

    native void unloadModel(Pointer handle);

    native String getModelType(Pointer handle);

    native int predictProba(
            Pointer handle, String text, int topK, String[] classes, float[] probabilities);

    native float[] getWordVector(Pointer handle, String word);

    native int runCmd(String[] args);
}
