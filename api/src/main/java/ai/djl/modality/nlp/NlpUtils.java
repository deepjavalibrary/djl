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
package ai.djl.modality.nlp;

import java.util.regex.Pattern;

/** Utility functions for processing String and Characters in NLP problems. */
public final class NlpUtils {

    private NlpUtils() {}

    /**
     * Check whether a character is is considered as a whitespace.
     *
     * <p>tab, newline and unicode space characters are all considered as whitespace.
     *
     * @param c input character to be checked.
     * @return whether a character is considered as a whitespace
     */
    public static boolean isWhiteSpace(char c) {
        return Character.isWhitespace(c) || Character.isSpaceChar(c);
    }

    /**
     * Check whether a character is is considered as a control character.
     *
     * <p>tab, newline and ios control characters are all considered as control character.
     *
     * @param c input character to be checked.
     * @return whether a character is considered as control character
     */
    public static boolean isControl(char c) {
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        return Character.isISOControl(c);
    }

    /**
     * Check whether a character is considered as a punctuation.
     *
     * <p>We treat all non-letter/number ASCII as punctuation. Characters such as "^", "$", and "`"
     * are not in the Unicode Punctuation class but we treat them as punctuation anyways, for
     * consistency.
     *
     * @param c input character to be checked
     * @return whether the character is considered as a punctuation
     */
    public static boolean isPunctuation(char c) {
        return Pattern.matches("[\\p{Punct}\\p{IsPunctuation}]", String.valueOf(c));
    }
}
