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

import java.util.Collections;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PunctuationSeparatorTest {

    @Test
    public void testPreprocess() {
        String sentence = "Hello.. How are you?!  @my_alias";
        String expected = "Hello . . How are you ? ! @ my _ alias";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        PunctuationSeparator punctuationSeparator = new PunctuationSeparator();
        List<String> processedTokens =
                punctuationSeparator.preprocess(Collections.singletonList(sentence));
        Assert.assertEquals(processedTokens.size(), 12);
        Assert.assertEquals(tokenizer.buildSentence(processedTokens), expected);
    }
}
