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

public class UnicodeNormalizerTest {

    @Test
    public void testDefaultNormalization() {
        final String sentence = "Unicode A\u0308A\uff21\uD835\uDC00\u00A01²₃\ufb01"; //contrary to checkstyle rules, we really need unicode escapes here, otherwise the test will easily break if reformatted
        final String expected = "Unicode \u00c4AAA 123fi"; //contrary to checkstyle rules, we really need unicode escapes here, otherwise the test will easily break if reformatted
        final SimpleTokenizer tokenizer = new SimpleTokenizer();
        final List<String> tokens = tokenizer.tokenize(sentence);
        final UnicodeNormalizer unicodeNormalizer = new UnicodeNormalizer();
        final List<String> processedTokens = unicodeNormalizer.preprocess(tokens);
        Assert.assertEquals(tokenizer.buildSentence(processedTokens), expected);
    }

}
