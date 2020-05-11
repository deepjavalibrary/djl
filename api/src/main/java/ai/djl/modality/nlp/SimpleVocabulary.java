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
     * Create a {@code SimpleVocabulary} object with ta {@link VocabularyBuilder}.
     *
     * @param builder the {@link VocabularyBuilder} to build the vocabulary with
     */
    public SimpleVocabulary(VocabularyBuilder builder) {
        reservedTokens = builder.reservedTokens;
        minFrequency = builder.minFrequency;
        unknownToken = builder.unknownToken;
        reservedTokens.add(unknownToken);
        for (List<String> sentence : builder.sentences) {
            addAllTokens(sentence);
        }
    }

    private void addToken(String token) {
        if (reservedTokens.contains(token)) {
            return;
        }
        TokenInfo tokenInfo = tokens.getOrDefault(token, new TokenInfo());
        if (++tokenInfo.frequency == minFrequency) {
            tokenInfo.index = indexToToken.size();
            indexToToken.add(token);
        }
        tokens.put(token, tokenInfo);
    }

    private void addAllTokens(Collection<String> tokens) {
        for (String token : tokens) {
            addToken(token);
        }
    }

    /**
     * Returns whether the given token is a known word.
     *
     * @param token the token
     * @return whether the given token is a known word
     */
    public boolean isKnownToken(String token) {
        if (reservedTokens.contains(token)) {
            return true;
        }
        if (!tokens.containsKey(token)) {
            return false;
        }
        return tokens.get(token).frequency >= minFrequency;
    }

    /**
     * Return a {@link String} used for unseen or rarely-seen tokens.
     *
     * @return the {@link String} used for unseen or rarely-seen tokens
     */
    public String getUnknownToken() {
        return unknownToken;
    }

    /**
     * Returns the token corresponding to the given index.
     *
     * @param index the index
     * @return the token corresponding to the given index
     */
    @Override
    public String getToken(long index) {
        if (index < 0 || index >= indexToToken.size()) {
            return unknownToken;
        }
        return indexToToken.get((int) index);
    }

    /**
     * Returns all the tokens in the vocabulary.
     *
     * @return the token corresponding to the given index
     */
    public List<String> getAllTokens() {
        Set<String> tokenSet = new HashSet<>(indexToToken);
        List<String> tokenList = new ArrayList<>();
        tokenList.addAll(reservedTokens);
        tokenSet.removeAll(reservedTokens);
        tokenList.addAll(tokenSet);
        return tokenList;
    }

    /**
     * Returns the index of the given token.
     *
     * @param token the token
     * @return the index of the given token
     */
    @Override
    public long getIndex(String token) {
        if (tokens.containsKey(token)) {
            return tokens.get(token).index;
        }
        return 0;
    }

    /**
     * Returns the size of the {@code Vocabulary}.
     *
     * @return the size of the {@code Vocabulary}
     */
    public int size() {
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
