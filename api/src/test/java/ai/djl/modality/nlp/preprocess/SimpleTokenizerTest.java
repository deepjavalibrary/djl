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

public class SimpleTokenizerTest {
    @Test
    public void testTokenize() {
        String sentence = "Hello! How are you?!";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        List<String> tokens = tokenizer.tokenize(sentence);
        String[] expectedTokens = new String[] {"Hello!", "How", "are", "you?!"};
        Assert.assertEquals(tokens.size(), expectedTokens.length);
        for (int i = 0; i < expectedTokens.length; i++) {
            Assert.assertEquals(tokens.get(i), expectedTokens[i]);
        }
        Assert.assertEquals(tokenizer.buildSentence(tokens), sentence);
    }

    @Test
    public void testTokenizeWithSingleWordToken() {
        String sentence = "Hello";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        List<String> tokens = tokenizer.tokenize(sentence);
        String[] expectedTokens = new String[] {"Hello"};
        Assert.assertEquals(tokens.size(), expectedTokens.length);
        for (int i = 0; i < expectedTokens.length; i++) {
            Assert.assertEquals(tokens.get(i), expectedTokens[i]);
        }
        Assert.assertEquals(tokenizer.buildSentence(tokens), sentence);
    }
}
