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
package ai.djl.modality.nlp;

import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;

/** The default implementation of Vocabulary. */
public class DefaultVocabulary implements Vocabulary {

    private Map<String, TokenInfo> tokens;
    private List<String> indexToToken;
    private Set<String> reservedTokens;
    private String unknownToken;

    /**
     * Creates a {@code DefaultVocabulary} object with the given list of tokens.
     *
     * @param tokens the {@link List} of tokens to build the vocabulary with
     */
    public DefaultVocabulary(List<String> tokens) {
        this(builder().add(tokens));
    }

    /**
     * Creates a {@code DefaultVocabulary} object with a {@link Builder}.
     *
     * @param builder the {@link Builder} to build the vocabulary with
     */
    public DefaultVocabulary(Builder builder) {
        tokens = new ConcurrentHashMap<>();
        reservedTokens = builder.reservedTokens;
        unknownToken = builder.unknownToken;
        if (unknownToken != null) {
            reservedTokens.add(unknownToken);
        }
        for (List<String> sentence : builder.sentences) {
            for (String token : sentence) {
                addToken(token);
            }
        }
        // Preserve order in vocab file, add reservedTokens after original vocab
        for (String token : reservedTokens) {
            addToken(token);
        }

        boolean pruned = pruneTokens(builder.minFrequency, builder.maxTokens);
        if (pruned) {
            initializeIndexToTokenReplacingIndices();
        } else {
            initializeIndexToTokenKeepingIndices();
        }
    }

    private void addToken(String token) {
        int index = tokens.size();
        tokens.compute(
                token,
                (k, v) -> {
                    if (v == null) {
                        v = new TokenInfo();
                        // Set index only when adding a new token
                        v.index = index;
                    }

                    // Update the frequency for both old and new tokens
                    if (reservedTokens.contains(k)) {
                        v.frequency = Integer.MAX_VALUE;
                    } else if (v.frequency < Integer.MAX_VALUE) {
                        ++v.frequency;
                    }
                    return v;
                });
    }

    /**
     * Removes tokens from {@code tokens} based on the arguments.
     *
     * @param minFrequency a minimum frequency where all tokens below it are pruned. -1 for no
     *     minFrequency.
     * @param maxSize a maximum number of tokens where only the maxSize most frequent are kept. -1
     *     for no maxSize
     * @return returns true if pruning occurred
     */
    private boolean pruneTokens(int minFrequency, int maxSize) {
        boolean pruned = false;
        // Prune tokens below min frequency
        if (minFrequency > 1) {
            for (Entry<String, TokenInfo> token : tokens.entrySet()) {
                if (token.getValue().frequency < minFrequency) {
                    tokens.remove(token.getKey());
                }
            }
            pruned = true;
        }

        // Prune number of tokens to maxSize
        if (maxSize > 0 && tokens.size() > maxSize) {
            tokens.entrySet()
                    .stream()
                    .sorted(
                            Map.Entry.comparingByValue(
                                    Comparator.comparingInt(
                                            tokenInfo ->
                                                    -tokenInfo.frequency))) // most frequent first
                    .skip(maxSize)
                    .forEach(token -> tokens.remove(token.getKey()));
            pruned = true;
        }
        return pruned;
    }

    /**
     * Initializes indexToToken using the indices in tokens.
     *
     * <p>This is used when not pruning to preserve the order tokens were given to this vocabulary.
     */
    private void initializeIndexToTokenKeepingIndices() {
        indexToToken = Arrays.asList(new String[tokens.size()]);
        for (Entry<String, TokenInfo> token : tokens.entrySet()) {
            indexToToken.set(Math.toIntExact(token.getValue().index), token.getKey());
        }
    }

    /**
     * Initializes indexToToken while replacing the indices in tokens.
     *
     * <p>When pruning, there will be unused indices. So, this will redo the indexing to be fully
     * compact without indexing gaps. The order of the original indices is preserved.
     */
    private void initializeIndexToTokenReplacingIndices() {
        indexToToken =
                tokens.entrySet()
                        .stream()
                        .sorted(Comparator.comparingLong(token -> token.getValue().index))
                        .map(Entry::getKey)
                        .collect(Collectors.toList());
        for (int i = 0; i < indexToToken.size(); i++) {
            tokens.get(indexToToken.get(i)).index = i;
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean contains(String token) {
        return tokens.containsKey(token);
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
            return tokens.get(token).index;
        }

        if (unknownToken != null) {
            return tokens.get(unknownToken).index;
        }

        throw new IllegalStateException(
                "Unexpected token in getIndex. Define an unknownToken for the vocabulary to enable support for unknown tokens.");
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return tokens.size();
    }

    /**
     * Creates a new builder to build a {@code DefaultVocabulary}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** Builder class that is used to build the {@link DefaultVocabulary}. */
    public static final class Builder {

        List<List<String>> sentences = new ArrayList<>();
        Set<String> reservedTokens = new HashSet<>();
        int minFrequency = -1;
        int maxTokens = -1;
        String unknownToken;

        private Builder() {}

        /**
         * Sets the optional parameter that specifies the minimum frequency to consider a token to
         * be part of the {@link DefaultVocabulary}. Defaults to no minimum.
         *
         * @param minFrequency the minimum frequency to consider a token to be part of the {@link
         *     DefaultVocabulary} or -1 for no minimum
         * @return this {@code VocabularyBuilder}
         */
        public Builder optMinFrequency(int minFrequency) {
            this.minFrequency = minFrequency;
            return this;
        }

        /**
         * Sets the optional limit on the size of the vocabulary.
         *
         * <p>The size includes the reservedTokens. If the number of added tokens exceeds the
         * maxToken limit, it keeps the most frequent tokens.
         *
         * @param maxTokens the maximum number of tokens or -1 for no maximum
         * @return this {@link Builder}
         */
        public Builder optMaxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        /**
         * Sets the optional parameter that specifies the unknown token's string value with
         * "&gt;unk&lt;".
         *
         * @return this {@code VocabularyBuilder}
         */
        public Builder optUnknownToken() {
            return optUnknownToken("<unk>");
        }

        /**
         * Sets the optional parameter that specifies the unknown token's string value.
         *
         * @param unknownToken the string value of the unknown token
         * @return this {@code VocabularyBuilder}
         */
        public Builder optUnknownToken(String unknownToken) {
            this.unknownToken = unknownToken;
            return this;
        }

        /**
         * Sets the optional parameter that sets the list of reserved tokens.
         *
         * @param reservedTokens the list of reserved tokens
         * @return this {@code VocabularyBuilder}
         */
        public Builder optReservedTokens(Collection<String> reservedTokens) {
            this.reservedTokens.addAll(reservedTokens);
            return this;
        }

        /**
         * Adds the given sentence to the {@link DefaultVocabulary}.
         *
         * @param sentence the sentence to be added
         * @return this {@code VocabularyBuilder}
         */
        public Builder add(List<String> sentence) {
            this.sentences.add(sentence);
            return this;
        }

        /**
         * Adds the given list of sentences to the {@link DefaultVocabulary}.
         *
         * @param sentences the list of sentences to be added
         * @return this {@code VocabularyBuilder}
         */
        public Builder addAll(List<List<String>> sentences) {
            this.sentences.addAll(sentences);
            return this;
        }

        /**
         * Adds a text vocabulary to the {@link DefaultVocabulary}.
         *
         * <pre>
         *   Example text file(vocab.txt):
         *   token1
         *   token2
         *   token3
         *   will be mapped to index of 0 1 2
         * </pre>
         *
         * @param path the path to the text file
         * @return this {@code VocabularyBuilder}
         * @throws IOException if failed to read vocabulary file
         */
        public Builder addFromTextFile(Path path) throws IOException {
            add(Utils.readLines(path, true));
            return this;
        }

        /**
         * Adds a text vocabulary to the {@link DefaultVocabulary}.
         *
         * @param url the text file url
         * @return this {@code VocabularyBuilder}
         * @throws IOException if failed to read vocabulary file
         */
        public Builder addFromTextFile(URL url) throws IOException {
            try (InputStream is = url.openStream()) {
                add(Utils.readLines(is, true));
            }
            return this;
        }

        /**
         * Adds a customized vocabulary to the {@link DefaultVocabulary}.
         *
         * @param url the text file url
         * @param lambda the function to parse the vocabulary file
         * @return this {@code VocabularyBuilder}
         */
        public Builder addFromCustomizedFile(URL url, Function<URL, List<String>> lambda) {
            return add(lambda.apply(url));
        }

        /**
         * Builds the {@link DefaultVocabulary} object with the set arguments.
         *
         * @return the {@link DefaultVocabulary} object built
         */
        public DefaultVocabulary build() {
            if (maxTokens > 0 && maxTokens < reservedTokens.size()) {
                throw new IllegalArgumentException(
                        "The vocabulary maxTokens can not be smaller than the number of reserved tokens");
            }
            return new DefaultVocabulary(this);
        }
    }

    /**
     * {@code TokenInfo} represents the information stored in the {@link DefaultVocabulary} about a
     * given token.
     */
    private static final class TokenInfo {
        int frequency;
        long index = -1;
    }
}
