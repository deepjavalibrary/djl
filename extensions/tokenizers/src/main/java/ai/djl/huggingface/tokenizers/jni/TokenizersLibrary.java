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
package ai.djl.huggingface.tokenizers.jni;

/** A class containing utilities to interact with the Tokenizer JNI layer. */
@SuppressWarnings("MissingJavadocMethod")
public final class TokenizersLibrary {

    public static final TokenizersLibrary LIB = new TokenizersLibrary();

    private TokenizersLibrary() {}

    public native long createTokenizer(String identifier);

    public native long createTokenizerFromString(String json);

    public native void deleteTokenizer(long handle);

    public native long encode(long tokenizer, String text, boolean addSpecialTokens);

    public native long encodeDual(
            long tokenizer, String text, String textPair, boolean addSpecialTokens);

    public native long encodeList(long tokenizer, String[] inputs, boolean addSpecialTokens);

    public native long[] batchEncode(long tokenizer, String[] inputs, boolean addSpecialTokens);

    public native long[] batchEncodePair(
            long tokenizer, String[] text, String[] textPair, boolean addSpecialTokens);

    public native String[] batchDecode(long tokenizer, long[][] batchIds, boolean addSpecialTokens);

    public native void deleteEncoding(long encoding);

    public native long[] getTokenIds(long encoding);

    public native long[] getTypeIds(long encoding);

    public native long[] getWordIds(long encoding);

    public native String[] getTokens(long encoding);

    public native long[] getAttentionMask(long encoding);

    public native long[] getSpecialTokenMask(long encoding);

    public native CharSpan[] getTokenCharSpans(long encoding);

    public native long[] getOverflowing(long encoding);

    public native String decode(long tokenizer, long[] ids, boolean addSpecialTokens);

    public native String getTruncationStrategy(long tokenizer);

    public native String getPaddingStrategy(long tokenizer);

    public native int getMaxLength(long tokenizer);

    public native int getStride(long tokenizer);

    public native int getPadToMultipleOf(long tokenizer);

    public native void disablePadding(long tokenizer);

    public native void setPadding(
            long tokenizer, int maxLength, String paddingStrategy, int padToMultipleOf);

    public native void disableTruncation(long tokenizer);

    public native void setTruncation(
            long tokenizer, int maxLength, String truncationStrategy, int stride);
}
