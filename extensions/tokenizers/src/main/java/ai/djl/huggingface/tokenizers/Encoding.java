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
package ai.djl.huggingface.tokenizers;

import ai.djl.huggingface.tokenizers.jni.CharSpan;

/** A class holds token encoding information. */
public class Encoding {

    private long[] ids;
    private long[] typeIds;
    private String[] tokens;
    private long[] wordIds;
    private long[] attentionMask;
    private long[] specialTokenMask;
    private CharSpan[] charTokenSpans;
    private Encoding[] overflowing;

    protected Encoding(
            long[] ids,
            long[] typeIds,
            String[] tokens,
            long[] wordIds,
            long[] attentionMask,
            long[] specialTokenMask,
            CharSpan[] charTokenSpans,
            Encoding[] overflowing) {
        this.ids = ids;
        this.typeIds = typeIds;
        this.tokens = tokens;
        this.wordIds = wordIds;
        this.attentionMask = attentionMask;
        this.specialTokenMask = specialTokenMask;
        this.charTokenSpans = charTokenSpans;
        this.overflowing = overflowing;
    }

    /**
     * Returns the token ids.
     *
     * @return the token ids
     */
    public long[] getIds() {
        return ids;
    }

    /**
     * Returns the token type ids.
     *
     * @return the token type ids
     */
    public long[] getTypeIds() {
        return typeIds;
    }

    /**
     * Returns the tokens.
     *
     * @return the tokens
     */
    public String[] getTokens() {
        return tokens;
    }

    /**
     * Returns the word ids.
     *
     * @return the word ids
     */
    public long[] getWordIds() {
        return wordIds;
    }

    /**
     * Returns the attention masks.
     *
     * @return the attention masks
     */
    public long[] getAttentionMask() {
        return attentionMask;
    }

    /**
     * Returns the special token masks.
     *
     * @return the special token masks
     */
    public long[] getSpecialTokenMask() {
        return specialTokenMask;
    }

    /**
     * Returns char token spans.
     *
     * @return char token spans
     */
    public CharSpan[] getCharTokenSpans() {
        return charTokenSpans;
    }

    /**
     * Returns an array of overflowing encodings.
     *
     * @return the array of overflowing encodings
     */
    public Encoding[] getOverflowing() {
        return overflowing;
    }
}
