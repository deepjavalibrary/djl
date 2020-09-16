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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Unicode normalization does not take care of "exotic" hyphens that we normally do not want in NLP
 * input. This preprocessor turns all Hyphens into "normal" ASCII minus-hyphen characters (U+002D).
 * Invisible soft hyphens are dropped from the input.
 */
public class HyphenNormalizer implements TextProcessor {

    private static final int SOFT_HYPHEN = 0x00AD;

    private static final Set<Integer> HYPHENS =
            new HashSet<>(
                    Arrays.asList(
                            0x002D, 0x007E, 0x00AD, 0x058A, 0x05BE, 0x2010, 0x2011, 0x2012, 0x2013,
                            0x2014, 0x2015, 0x2053, 0x207B, 0x208B, 0x2212, 0x2E3A, 0x2E3B, 0x301C,
                            0x3030, 0xFE31, 0xFE32, 0xFE58, 0xFE63, 0xFF0D));

    /**
     * Returns whether the given code point is a hyphen-like codepoint. Tests for hyphen-minus,
     * tilde, soft hyphen, armenian hyphen, hebrew punctuation maqaf, canadian syllabics hyphen,
     * mongolian hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal bar, swung
     * dash, superscript minus, subscript minus, minus sign, double oblique hyphen, two-em dash,
     * three-em dash, wave dash, wavy dash, katakana-hiragana double hyphen
     *
     * @param codePoint A unicode code point. (not a char!)
     * @return true: given code point represents a hyphen-like glyph
     */
    public static boolean isHyphenLike(final Integer codePoint) {
        return HYPHENS.contains(codePoint);
    }

    /**
     * Replaces hyphen like codepoints by ASCII "-", removes soft hyphens.
     *
     * @param s input string to replace hyphens in
     * @return the same string with soft hyphens dropped and hyphen-like codepoints replaced by an
     *     ASCII minus.
     */
    public static String normalizeHyphens(final String s) {
        final StringBuilder temp = new StringBuilder(s.length());
        int position = 0;
        while (position < s.length()) {
            final int cp = s.codePointAt(position);
            if (cp == SOFT_HYPHEN) { // drop soft hyphens
                // do nothing
            } else if (isHyphenLike(cp)) { // replace "exotic" hyphens by a simple ASCII '-'
                temp.append('-');
            } else {
                temp.appendCodePoint(cp);
            }
            position += Character.isBmpCodePoint(cp) ? 1 : 2;
        }
        return temp.toString();
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(final List<String> tokens) {
        return tokens.stream().map(HyphenNormalizer::normalizeHyphens).collect(Collectors.toList());
    }
}
