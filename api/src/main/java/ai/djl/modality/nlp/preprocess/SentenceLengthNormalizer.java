/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp.preprocess;

import java.util.ArrayList;
import java.util.List;

/**
 * {@code SentenceLengthNormalizer} normalizes the length of all the input sentences to the
 * specified number of tokens.
 *
 * <p>If the number of tokens in the input sentence is higher than the fixed length, the sentence is
 * truncated to the fixed number. If the number of tokens in the input sentence is fewer than the
 * fixed sentence length, padding tokens are inserted to make the length equal to the sentence
 * length.
 */
public class SentenceLengthNormalizer implements TextProcessor {

    private static final int DEFAULT_SENTENCE_LENGTH = 10;
    private static final String DEFAULT_PADDING_TOKEN = "<pad>";
    private static final String DEFAULT_EOS_TOKEN = "<eos>";
    private static final String DEFAULT_BOS_TOKEN = "<bos>";

    private int sentenceLength;
    private boolean addEosBosTokens;
    private String paddingToken;
    private String eosToken;
    private String bosToken;
    private int lastValidLength = -1;

    /** Creates a {@link TextProcessor} that normalizes the length of the input. */
    public SentenceLengthNormalizer() {
        this(DEFAULT_SENTENCE_LENGTH, false);
    }

    /**
     * Creates a {@link TextProcessor} that normalizes the length of the input to the given sentence
     * length.
     *
     * @param sentenceLength the sentence length
     * @param addEosBosTokens whether to add Eos and Bos tokens before normalizing sentence length
     */
    public SentenceLengthNormalizer(int sentenceLength, boolean addEosBosTokens) {
        this(
                sentenceLength,
                addEosBosTokens,
                DEFAULT_PADDING_TOKEN,
                DEFAULT_EOS_TOKEN,
                DEFAULT_BOS_TOKEN);
    }

    /**
     * Creates a {@link TextProcessor} that normalizes the length of the input to the given sentence
     * length.
     *
     * @param sentenceLength the sentence length
     * @param addEosBosTokens whether to add Eos and Bos tokens before normalizing sentence length
     * @param paddingToken the padding token to be used if the number of tokens in the input is less
     *     than sentence length
     * @param eosToken the end of sentence token
     * @param bosToken the begining of sentence token
     */
    public SentenceLengthNormalizer(
            int sentenceLength,
            boolean addEosBosTokens,
            String paddingToken,
            String eosToken,
            String bosToken) {
        this.sentenceLength = sentenceLength;
        this.addEosBosTokens = addEosBosTokens;
        this.paddingToken = paddingToken;
        this.eosToken = eosToken;
        this.bosToken = bosToken;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(List<String> tokens) {
        List<String> list = new ArrayList<>(sentenceLength);
        if (addEosBosTokens) {
            list.add(bosToken);
        }
        list.addAll(tokens);
        if (addEosBosTokens) {
            list.add(eosToken);
        }
        int size = list.size();
        if (sentenceLength < size) {
            lastValidLength = sentenceLength;
            if (addEosBosTokens) {
                list.set(sentenceLength - 1, eosToken);
            }
            return list.subList(0, sentenceLength);
        }
        lastValidLength = size;
        for (int i = size; i < sentenceLength; ++i) {
            list.add(paddingToken);
        }
        return list;
    }

    /**
     * Returns the valid length of the sentence that was last served as input to {@link
     * SentenceLengthNormalizer#preprocess(List)}. If no sentences preprocess before calling this
     * method, it will -1.
     *
     * @return the valid length of the sentence that was last preprocessed
     */
    public int getLastValidLength() {
        return lastValidLength;
    }
}
