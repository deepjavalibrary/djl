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
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * {@code LowerCaseConvertor} converts every character of the input tokens to it's respective lower
 * case character.
 */
public class LowerCaseConvertor implements TextProcessor {

    private static final Locale DEFAULT_LOCALE = Locale.ENGLISH;

    private final Locale locale;

    /**
     * Creates a {@link TextProcessor} that converts input text into lower case character given the
     * {@link Locale}.
     *
     * @param locale the expected {@link Locale} of the input text
     */
    public LowerCaseConvertor(Locale locale) {
        this.locale = locale;
    }

    /**
     * Creates a {@link TextProcessor} that converts input text into lower case character with the
     * default english {@link Locale}.
     */
    public LowerCaseConvertor() {
        this(DEFAULT_LOCALE);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(List<String> tokens) {
        return tokens.stream().map(s -> s.toLowerCase(locale)).collect(Collectors.toList());
    }
}
