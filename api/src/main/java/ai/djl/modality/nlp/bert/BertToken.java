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
package ai.djl.modality.nlp.bert;

import java.util.List;

/** BertToken contains all the information for Bert model after encoding question and paragraph. */
public class BertToken {
    private List<String> tokens;
    private List<Long> tokenType;
    private List<Long> attentionMask;
    private int validLength;

    /**
     * Creates an instance of BertToken which includes information for Bert model.
     *
     * @param tokens indices of input sequence tokens in the vocabulary.
     * @param tokenType segment token indices to indicate first and second portions of the inputs.
     * @param attentionMask mask to avoid performing attention on padding token indices.
     * @param validLength length that indicates the original input sequence.
     */
    public BertToken(
            List<String> tokens, List<Long> tokenType, List<Long> attentionMask, int validLength) {
        this.tokens = tokens;
        this.tokenType = tokenType;
        this.attentionMask = attentionMask;
        this.validLength = validLength;
    }

    /**
     * Gets the indices of input sequence tokens in the vocabulary.
     *
     * @return indices of input sequence tokens
     */
    public List<String> getTokens() {
        return tokens;
    }

    /**
     * Gets segment token indices to indicate first and second portions of the inputs.
     *
     * @return segment token indices
     */
    public List<Long> getTokenTypes() {
        return tokenType;
    }

    /**
     * Gets the mask to avoid performing attention on padding token indices.
     *
     * @return mask that performs attention on non-padding token indices
     */
    public List<Long> getAttentionMask() {
        return attentionMask;
    }

    /**
     * Gets the length of the original sentence which has question and paragraph.
     *
     * @return length of the original sentence which has question and paragraph
     */
    public int getValidLength() {
        return validLength;
    }
}
