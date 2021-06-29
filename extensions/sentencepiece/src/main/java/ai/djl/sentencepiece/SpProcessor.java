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
import ai.djl.sentencepiece.jni.SentencePieceLibrary;
import ai.djl.util.NativeResource;

/** The processor holder for SentencePiece. */
public final class SpProcessor extends NativeResource<Long> {

    private static RuntimeException libraryStatus;

    static {
        try {
            LibUtils.loadLibrary();
        } catch (RuntimeException e) {
            libraryStatus = e;
        }
    }

    private SpProcessor() {
        super(SentencePieceLibrary.LIB.createSentencePieceProcessor());
    }

    public static SpProcessor newInstance() {
        if (libraryStatus != null) {
            throw libraryStatus;
        }
        return new SpProcessor();
    }

    public void loadModel(String path) {
        SentencePieceLibrary.LIB.loadModel(getHandle(), path);
    }

    public String[] tokenize(String input) {
        return SentencePieceLibrary.LIB.tokenize(getHandle(), input);
    }

    public String buildSentence(String[] tokens) {
        return SentencePieceLibrary.LIB.detokenize(getHandle(), tokens);
    }

    public String getToken(int id) {
        return SentencePieceLibrary.LIB.idToPiece(getHandle(), id);
    }

    public int getId(String token) {
        return SentencePieceLibrary.LIB.pieceToId(getHandle(), token);
    }

    public int[] encode(String sentence) {
        return SentencePieceLibrary.LIB.encode(getHandle(), sentence);
    }

    public String decode(int[] ids) {
        return SentencePieceLibrary.LIB.decode(getHandle(), ids);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.get();
        if (pointer != null) {
            SentencePieceLibrary.LIB.deleteSentencePieceProcessor(pointer);
        }
    }
}
