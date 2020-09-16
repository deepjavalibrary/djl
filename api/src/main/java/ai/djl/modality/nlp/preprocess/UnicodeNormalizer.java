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

import java.text.Normalizer;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Applies unicode normalization to input strings. This is particularly important if you are dealing
 * with non-English input or with text originating from OCR applications.
 */
public class UnicodeNormalizer implements TextProcessor {

    public static final Normalizer.Form DEFAULT_FORM = Normalizer.Form.NFKC;

    private final Normalizer.Form normalForm;

    /**
     * Unicode normalizer with a configurable normal form.
     *
     * @param normalForm The normal form to use.
     */
    public UnicodeNormalizer(final Normalizer.Form normalForm) {
        this.normalForm = normalForm;
    }

    /**
     * Default version of the Unicode Normalizer using NFKC normal form. If you do not know what
     * normal form you need, this is the normal form you need.
     */
    public UnicodeNormalizer() {
        this(DEFAULT_FORM);
    }

    /**
     * Normalizes a String using a sensible default normal form. Use this if you do not want to
     * think about unicode preprocessing.
     *
     * @param s Any non-null string
     * @return The given string with default unicode normalization applied.
     */
    public static String normalizeDefault(final String s) {
        return Normalizer.normalize(s, DEFAULT_FORM);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(final List<String> tokens) {
        return tokens.stream()
                .map((s) -> Normalizer.normalize(s, normalForm))
                .collect(Collectors.toList());
    }
}
