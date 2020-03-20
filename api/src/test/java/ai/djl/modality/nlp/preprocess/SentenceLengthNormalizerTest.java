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

public class SentenceLengthNormalizerTest {
    @Test
    public void testPreprocessForShortSentence() {
        String sentence = "Hello.. How are you?!";
        String expectedSentence = "Hello . . How are you ? ! <pad> <pad>";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        List<String> tokens = tokenizer.tokenize(sentence);
        PunctuationSeparator punctuationSeparator = new PunctuationSeparator();
        tokens = punctuationSeparator.preprocess(tokens);
        tokens = new SentenceLengthNormalizer().preprocess(tokens);
        String[] expectedTokens =
                new String[] {"Hello", ".", ".", "How", "are", "you", "?", "!", "<pad>", "<pad>"};
        Assert.assertEquals(tokens.toArray(), expectedTokens);
        Assert.assertEquals(tokenizer.buildSentence(tokens), expectedSentence);
    }

    @Test
    public void testPreprocess() {
        String sentence = "The quick brown fox jumps over the lazy dog.";
        String expectedSentence = "The quick brown fox jumps over the lazy dog .";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        List<String> tokens = tokenizer.tokenize(sentence);
        PunctuationSeparator punctuationSeparator = new PunctuationSeparator();
        tokens = punctuationSeparator.preprocess(tokens);
        tokens = new SentenceLengthNormalizer().preprocess(tokens);
        String[] expectedTokens =
                new String[] {
                    "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."
                };
        Assert.assertEquals(tokens.toArray(), expectedTokens);
        Assert.assertEquals(tokenizer.buildSentence(tokens), expectedSentence);
    }

    @Test
    public void testPreprocessForTruncation() {
        String sentence = "The quick brown fox jumps over the lazy dog.";
        String expectedSentence = "The quick brown fox jumps over the lazy";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        List<String> tokens = tokenizer.tokenize(sentence);
        tokens = new SentenceLengthNormalizer(8, false).preprocess(tokens);
        String[] expectedTokens =
                new String[] {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy"};
        Assert.assertEquals(tokens.toArray(), expectedTokens);
        Assert.assertEquals(tokenizer.buildSentence(tokens), expectedSentence);
    }

    @Test
    public void testPreprocessWithEosBosForTruncation() {
        String sentence = "The quick brown fox jumps over the lazy dog.";
        String expectedSentence = "<bos> The quick brown fox jumps over the lazy <eos>";
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        List<String> tokens = tokenizer.tokenize(sentence);
        tokens = new SentenceLengthNormalizer(10, true).preprocess(tokens);
        String[] expectedTokens =
                new String[] {
                    "<bos>", "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "<eos>"
                };
        Assert.assertEquals(tokens.toArray(), expectedTokens);
        Assert.assertEquals(tokenizer.buildSentence(tokens), expectedSentence);
    }
}
