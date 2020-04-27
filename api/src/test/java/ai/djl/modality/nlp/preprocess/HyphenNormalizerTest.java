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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.List;

public class HyphenNormalizerTest {
    @Test
    public void testHyphenNormalization() {
        final String hyphens1 = "-\u002D\u007E\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2053"; //contrary to checkstyle rules, we really need unicode escapes here, otherwise the test will easily break if reformatted
        final String softHyphen = " Uni\u00adcode "; //contrary to checkstyle rules, we really need unicode escapes here, otherwise the test will easily break if reformatted
        final String hyphens2 = "\u207B\u208B\u2212\u2E3A\u2E3B\u301C\u3030\uFE31\uFE32\uFE58\uFE63\uFF0D"; //contrary to checkstyle rules, we really need unicode escapes here, otherwise the test will easily break if reformatted
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
