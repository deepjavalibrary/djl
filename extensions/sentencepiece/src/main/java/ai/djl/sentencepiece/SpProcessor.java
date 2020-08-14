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
package ai.djl.sentencepiece;

import ai.djl.sentencepiece.jni.LibUtils;
import ai.djl.sentencepiece.jni.NativeResource;
import ai.djl.sentencepiece.jni.SentencePieceLibrary;

/** The processor holder for SentencePiece. */
class SpProcessor extends NativeResource {

    static {
        LibUtils.loadLibrary();
    }

    SpProcessor() {
        super(SentencePieceLibrary.LIB.createSentencePieceProcessor());
    }

    void loadModel(String path) {
        SentencePieceLibrary.LIB.loadModel(getHandle(), path);
    }

    String[] tokenize(String input) {
        return SentencePieceLibrary.LIB.tokenize(getHandle(), input);
    }

    String buildSentence(String[] tokens) {
        return SentencePieceLibrary.LIB.detokenize(getHandle(), tokens);
    }

    String getToken(int id) {
        return SentencePieceLibrary.LIB.idToPiece(getHandle(), id);
    }

    int getId(String token) {
        return SentencePieceLibrary.LIB.pieceToId(getHandle(), token);
    }

    @Override
    public void close() {
        SentencePieceLibrary.LIB.deleteSentencePieceProcessor(getHandle());
    }
}
