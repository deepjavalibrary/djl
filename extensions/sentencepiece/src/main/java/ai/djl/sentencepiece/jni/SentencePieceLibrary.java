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
package ai.djl.sentencepiece.jni;

/** A class containing utilities to interact with the SentencePiece Engine's JNI layer. */
@SuppressWarnings("MissingJavadocMethod")
public final class SentencePieceLibrary {

    public static final SentencePieceLibrary LIB = new SentencePieceLibrary();

    private SentencePieceLibrary() {}

    public native Pointer createSentencePieceProcessor();

    public native void loadModel(Pointer handle, String filePath);

    public native void deleteSentencePieceProcessor(Pointer handle);

    public native String[] tokenize(Pointer handle, String text);

    public native int[] encode(Pointer handle, String text);

    public native String detokenize(Pointer handle, String[] tokens);

    public native String decode(Pointer handle, int[] ids);

    public native String idToPiece(Pointer handle, int id);

    public native int pieceToId(Pointer handle, String piece);
}
