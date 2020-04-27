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
public class HyphenNormalizerTest {
    @Test
    public void testHyphenNormalization() {
        // the following are a number of hyphen like glyphs from various languages
        // and various special hyphens (like the half width hyphen)
        final String hyphens1 =
                "-\u002D\u007E\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2053";
        // the soft-hyphen is only meant to indicate were words can be split
        // and should be removed in NLP applications
        final String softHyphen = " Uni\u00adcode ";
        // ...and more hyphens
        final String hyphens2 =
                "\u207B\u208B\u2212\u2E3A\u2E3B\u301C\u3030\uFE31\uFE32\uFE58\uFE63\uFF0D";
        final String sentence = hyphens1 + softHyphen + hyphens2;
        final String expected = "------------ Unicode ------------";
        final SimpleTokenizer tokenizer = new SimpleTokenizer();
        final List<String> tokens = tokenizer.tokenize(sentence);
        final HyphenNormalizer hyphenNormalizer = new HyphenNormalizer();
        final List<String> processedTokens = hyphenNormalizer.preprocess(tokens);
        Assert.assertEquals(tokenizer.buildSentence(processedTokens), expected);
    }

    public static void main(final String[] args) {
        new HyphenNormalizerTest().testHyphenNormalization();
    }
}
