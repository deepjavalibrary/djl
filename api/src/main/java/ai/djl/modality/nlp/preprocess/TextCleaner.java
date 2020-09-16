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
package ai.djl.modality.nlp.preprocess;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/** Applies remove or replace of certain characters based on condition. */
public class TextCleaner implements TextProcessor {
    private Function<Character, Boolean> condition;
    private char replace;

    /**
     * Remove a character if it meets the condition supplied.
     *
     * @param condition lambda function that defines whether a character meets condition
     */
    public TextCleaner(Function<Character, Boolean> condition) {
        this.condition = condition;
    }

    /**
     * Replace a character if it meets the condition supplied.
     *
     * @param condition lambda function that defines whether a character meets condition
     * @param replace the character to replace
     */
    public TextCleaner(Function<Character, Boolean> condition, char replace) {
        this.condition = condition;
        this.replace = replace;
    }

    private String cleanText(String text) {
        StringBuilder sb = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (condition.apply(c)) {
                if (replace == '\u0000') {
                    continue;
                } else {
                    sb.append(replace);
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(List<String> tokens) {
        return tokens.stream().map(this::cleanText).collect(Collectors.toList());
    }
}
