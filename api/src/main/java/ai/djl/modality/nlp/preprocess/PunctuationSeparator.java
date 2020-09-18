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

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** {@code PunctuationSeparator} separates punctuation into a separate token. */
public class PunctuationSeparator implements TextProcessor {
    private static final Pattern PATTERN =
            Pattern.compile(
                    "\\s+|(?<=[\\p{Punct}\\p{IsPunctuation}])|(?=[\\p{Punct}\\p{IsPunctuation}])");

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(List<String> tokens) {
        return tokens.stream()
                .map(PATTERN::split)
                .flatMap(Arrays::stream)
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());
    }
}
