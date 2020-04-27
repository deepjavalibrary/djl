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
import org.testng.Assert;
import org.testng.annotations.Test;

// We need to test specific unicode char changes - however some editors tend
// to perform unicode normalization while saving, copy-pasting etc.
// Hence this test would be broken accidentally if we were not using escaped unicode chars.
// But in order for the build to pass, we need to suppress we need to suppress the checkstyle rule
// here.
@SuppressWarnings("AvoidEscapedUnicodeCharacters")
public class UnicodeNormalizerTest {

    @Test
    public void testDefaultNormalization() {
        // the following tests are contained in the test string below:
        // - Umlaut Ä (with separate diaresis, two chars for one German letter,
        // causes German dictionary lookups to fail)
        // - "normal" A, should stay as is
        // - full width capital latin A, occurs in mixed CJK/Latin
        // text sources, should be turned into a normal A
        // - Bold math letter A, not only used in math texts, but also
        // often used in social media (e.g. twitter) to fake formatting,
        // should be turned into a normal A as well
        // - non breaking space, should be turned into a normal space
        // - 1,2,3 in normal, super- and sub-script
        // - an fi ligature, should be turned into f and i
        final String sentence = "Unicode A\u0308A\uff21\uD835\uDC00\u00A01²₃\ufb01";
        final String expected = "Unicode \u00c4AAA 123fi";
        final SimpleTokenizer tokenizer = new SimpleTokenizer();
        final List<String> tokens = tokenizer.tokenize(sentence);
        final UnicodeNormalizer unicodeNormalizer = new UnicodeNormalizer();
        final List<String> processedTokens = unicodeNormalizer.preprocess(tokens);
        Assert.assertEquals(tokenizer.buildSentence(processedTokens), expected);
    }
}
