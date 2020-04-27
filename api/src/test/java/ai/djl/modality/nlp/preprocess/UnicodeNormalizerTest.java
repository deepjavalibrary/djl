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
import org.testng.Assert;
import org.testng.annotations.Test;

public class UnicodeNormalizerTest {

    @Test
    public void testDefaultNormalization() {
        // Contrary to checkstyle rules, we really need unicode escapes here,
        // otherwise the test will easily break if reformatted or depending
        // on the behaviour of the editor using to edit this class.
        // To get around checkstyle, we need a comment (any will do)  at the
        // end of the line to supress the error after the string constants.
        // These however break the build formatting check when run through the
        // autoformatter if they are too long....
        // Hence there is a dummy comment a the end of the literals to suppress
        // checkstyle that is short enough to  make it through the autoformat
        // unscathed. DO NOT REMOVE THOSE, OTHERWISE THE BUILD BREAKS!
        final String sentence = "Unicode A\u0308A\uff21\uD835\uDC00\u00A01²₃\ufb01"; // suppress
        final String expected = "Unicode \u00c4AAA 123fi"; // suppress
        final SimpleTokenizer tokenizer = new SimpleTokenizer();
        final List<String> tokens = tokenizer.tokenize(sentence);
        final UnicodeNormalizer unicodeNormalizer = new UnicodeNormalizer();
        final List<String> processedTokens = unicodeNormalizer.preprocess(tokens);
        Assert.assertEquals(tokenizer.buildSentence(processedTokens), expected);
    }
}
