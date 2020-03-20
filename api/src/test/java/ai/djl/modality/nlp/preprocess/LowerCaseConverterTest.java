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

public class LowerCaseConverterTest {
    @Test
    public void testPreprocess() {
        String sentence = "Hello. How are you?!";
        String expected = "hello. how are you?!";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        List<String> tokens = tokenizer.tokenize(sentence);
        LowerCaseConvertor lowerCaseConvertor = new LowerCaseConvertor();
        List<String> processedTokens = lowerCaseConvertor.preprocess(tokens);
        Assert.assertEquals(tokenizer.buildSentence(processedTokens), expected);
    }
}
