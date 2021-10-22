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

import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** BertTokenizer is a class to help you encode question and paragraph sentence. */
public class BertTokenizer extends SimpleTokenizer {

    private static final Pattern PATTERN = Pattern.compile("(\\S+?)([.,?!])?(\\s+|$)");

    /** {@inheritDoc} */
    @Override
    public List<String> tokenize(String input) {
        List<String> ret = new LinkedList<>();

        Matcher m = PATTERN.matcher(input);
        while (m.find()) {
            ret.add(m.group(1));
            String token = m.group(2);
            if (token != null) {
                ret.add(token);
            }
        }

        return ret;
    }

    /**
     * Returns a string presentation of the tokens.
     *
     * @param tokens a list of tokens
     * @return a string presentation of the tokens
     */
    public String tokenToString(List<String> tokens) {
        return String.join(" ", tokens);
    }

    /**
     * Pads the tokens to the required length.
     *
     * @param <E> the type of the List
     * @param tokens the input tokens
     * @param padItem the things to pad at the end
     * @param num the total length after padding
     * @return a list of padded tokens
     */
    public <E> List<E> pad(List<E> tokens, E padItem, int num) {
        if (tokens.size() >= num) {
            return tokens;
        }
        List<E> padded = new ArrayList<>(num);
        padded.addAll(tokens);
        for (int i = tokens.size(); i < num; ++i) {
            padded.add(padItem);
        }
        return padded;
    }

    /**
     * Encodes questions and paragraph sentences.
     *
     * @param question the input question
     * @param paragraph the input paragraph
     * @return BertToken
     */
    public BertToken encode(String question, String paragraph) {
        List<String> qToken = tokenize(question);
        List<String> pToken = tokenize(paragraph);
        int validLength = qToken.size() + pToken.size();
        qToken.add(0, "[CLS]");
        qToken.add("[SEP]");
        pToken.add("[SEP]");
        List<String> tokens = new ArrayList<>(qToken);
        tokens.addAll(pToken);

        int tokenTypeStartIdx = qToken.size();
        long[] tokenTypeArr = new long[tokens.size()];
        Arrays.fill(tokenTypeArr, tokenTypeStartIdx, tokenTypeArr.length, 1);

        long[] attentionMaskArr = new long[tokens.size()];
        Arrays.fill(attentionMaskArr, 1);

        return new BertToken(
                tokens,
                Arrays.stream(tokenTypeArr).boxed().collect(Collectors.toList()),
                Arrays.stream(attentionMaskArr).boxed().collect(Collectors.toList()),
                validLength);
    }

    /**
     * Encodes questions and paragraph sentences with max length.
     *
     * @param question the input question
     * @param paragraph the input paragraph
     * @param maxLength the maxLength
     * @return BertToken
     */
    public BertToken encode(String question, String paragraph, int maxLength) {
        BertToken bertToken = encode(question, paragraph);
        return new BertToken(
                pad(bertToken.getTokens(), "[PAD]", maxLength),
                pad(bertToken.getTokenTypes(), 0L, maxLength),
                pad(bertToken.getAttentionMask(), 0L, maxLength),
                bertToken.getValidLength());
    }
}
