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
package ai.djl.modality.nlp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** The simple implementation of Vocabulary. */
public class SimpleVocabulary implements Vocabulary {

    private Map<String, TokenInfo> tokens = new ConcurrentHashMap<>();
    private List<String> indexToToken = new ArrayList<>();
    private Set<String> reservedTokens;
    private int minFrequency;
    private String unknownToken;

    /**
     * Create a {@code SimpleVocabulary} object with a {@link VocabularyBuilder}.
     *
     * @param builder the {@link VocabularyBuilder} to build the vocabulary with
     */
    public SimpleVocabulary(VocabularyBuilder builder) {
        reservedTokens = builder.reservedTokens;
        minFrequency = builder.minFrequency;
        unknownToken = builder.unknownToken;
        reservedTokens.add(unknownToken);
        addTokens(reservedTokens);
        for (List<String> sentence : builder.sentences) {
            for (String word : sentence) {
                addWord(word);
            }
        }
    }

    /**
     * Create a {@code SimpleVocabulary} object with the given list of tokens.
     *
     * @param tokens the {@link List} of tokens to build the vocabulary with
     */
    public SimpleVocabulary(List<String> tokens) {
        reservedTokens = new HashSet<>();
        minFrequency = 10;
        unknownToken = "<unk>";
        reservedTokens.add(unknownToken);
        addTokens(reservedTokens);
        addTokens(tokens);
    }

    private void addWord(String token) {
        if (reservedTokens.contains(token)) {
            return;
        }
        TokenInfo tokenInfo = tokens.getOrDefault(token, new TokenInfo());
        if (++tokenInfo.frequency == minFrequency) {
            tokenInfo.index = tokens.size();
            indexToToken.add(token);
        }
        tokens.put(token, tokenInfo);
    }

    private void addTokens(Collection<String> tokens) {
        for (String token : tokens) {
            TokenInfo tokenInfo = new TokenInfo();
            tokenInfo.frequency = Integer.MAX_VALUE;
            tokenInfo.index = indexToToken.size();
            indexToToken.add(token);
            this.tokens.put(token, tokenInfo);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String getToken(long index) {
        if (index < 0 || index >= indexToToken.size()) {
            return unknownToken;
        }
        return indexToToken.get((int) index);
    }

    /** {@inheritDoc} */
    @Override
    public long getIndex(String token) {
        if (tokens.containsKey(token)) {
            TokenInfo tokenInfo = tokens.get(token);
            if (tokenInfo.frequency >= minFrequency) {
                return tokenInfo.index;
            }
        }
        return tokens.get(unknownToken).index;
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return tokens.size();
    }

    /** Builder class that is used to build the {@link SimpleVocabulary}. */
    public static class VocabularyBuilder {
        protected List<List<String>> sentences = new LinkedList<>();
        protected Set<String> reservedTokens = new HashSet<>();
        protected int minFrequency = 10;
        protected String unknownToken = "<unk>";

        /**
         * Sets the optional parameter that specifies the minimum frequency to consider a token to
         * be part of the {@link SimpleVocabulary}. Defaults to 10.
         *
         * @param minFrequency the minimum frequency to consider a token to be part of the {@link
         *     SimpleVocabulary}
         * @return this {@code VocabularyBuilder}
         */
        public VocabularyBuilder optMinFrequency(int minFrequency) {
            this.minFrequency = minFrequency;
            return this;
        }

        /**
         * Sets the optional parameter that specifies the unknown token's string value.
         *
         * @param unknownToken the string value of the unknown token
         * @return this {@code VocabularyBuilder}
         */
        public VocabularyBuilder optUnknownToken(String unknownToken) {
            this.unknownToken = unknownToken;
            return this;
        }

        /**
         * Sets the optional parameter that sets the list of reserved tokens.
         *
         * @param reservedTokens the list of reserved tokens
         * @return this {@code VocabularyBuilder}
         */
        public VocabularyBuilder optReservedTokens(Collection<String> reservedTokens) {
            this.reservedTokens.addAll(reservedTokens);
            return this;
        }

        /**
         * Adds the given sentence to the {@link SimpleVocabulary}.
         *
         * @param sentence the sentence to be added
         * @return this {@code VocabularyBuilder}
         */
        public VocabularyBuilder add(List<String> sentence) {
            this.sentences.add(sentence);
            return this;
        }

        /**
         * Adds the given list of sentences to the {@link SimpleVocabulary}.
         *
         * @param sentences the list of sentences to be added
         * @return this {@code VocabularyBuilder}
         */
        public VocabularyBuilder addAll(List<List<String>> sentences) {
            this.sentences.addAll(sentences);
            return this;
        }

        /**
         * Builds the {@link SimpleVocabulary} object with the set arguments.
         *
         * @return the {@link SimpleVocabulary} object built
         */
        public SimpleVocabulary build() {
            return new SimpleVocabulary(this);
        }
    }

    /**
     * {@code TokenInfo} represents the information stored in the {@link SimpleVocabulary} about a
     * given token.
     */
    private static final class TokenInfo {
        int frequency;
        long index = -1;

        public TokenInfo() {}
    }
}
