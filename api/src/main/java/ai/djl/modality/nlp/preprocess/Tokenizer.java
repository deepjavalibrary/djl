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

import java.util.List;
import java.util.stream.Collectors;

/**
 * {@code Tokenizer} interface provides the ability to break-down sentences into embeddable tokens.
 */
public interface Tokenizer extends TextProcessor {

    /** {@inheritDoc} */
    @Override
    default List<String> preprocess(List<String> tokens) {
        return tokens.stream()
                .map(this::tokenize)
                .flatMap(List::stream)
                .collect(Collectors.toList());
    }

    /**
     * Breaks down the given sentence into a list of tokens that can be represented by embeddings.
     *
     * @param sentence the sentence to tokenize
     * @return a {@link List} of tokens
     */
    List<String> tokenize(String sentence);

    /**
     * Combines a list of tokens to form a sentence.
     *
     * @param tokens the {@link List} of tokens
     * @return the sentence built from the given tokens
     */
    String buildSentence(List<String> tokens);
}
