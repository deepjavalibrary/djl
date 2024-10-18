/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.huggingface.tokenizers;

import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.jni.CharSpan;
import ai.djl.testing.TestRequirements;
import ai.djl.training.util.DownloadUtils;
import ai.djl.util.PairList;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class HuggingFaceTokenizerTest {

    @Test
    public void testVersion() {
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            String djlVersion = Engine.getDjlVersion();
            Assert.assertEquals(tokenizer.getVersion(), "0.20.0-" + djlVersion);
        }
    }

    @Test
    public void testTokenizer() throws IOException {
        String input = "Hello, y'all! How are you 😁 ?";
        String[] inputs = {"Hello, y'all!", "How are you 😁 ?"};

        String[] expected = {
            "[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"
        };

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder().optTokenizerName("bert-base-cased").build()) {
            Assert.assertEquals(tokenizer.getTruncation(), "LONGEST_FIRST");
            Assert.assertEquals(tokenizer.getPadding(), "LONGEST");
            Assert.assertEquals(tokenizer.getMaxLength(), 512);
            Assert.assertEquals(tokenizer.getStride(), 0);
            Assert.assertEquals(tokenizer.getPadToMultipleOf(), 0);

            List<String> ret = tokenizer.tokenize(input);
            Assert.assertEquals(ret.toArray(Utils.EMPTY_ARRAY), expected);
            Encoding encoding = tokenizer.encode(input);

            long[] ids = {101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 100, 136, 102};
            long[] typeIds = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            long[] wordIds = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1};
            long[] attentionMask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            long[] specialTokenMask = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

            Assert.assertEquals(expected, encoding.getTokens());
            Assert.assertEquals(ids, encoding.getIds());
            Assert.assertEquals(typeIds, encoding.getTypeIds());
            Assert.assertEquals(wordIds, encoding.getWordIds());
            Assert.assertEquals(attentionMask, encoding.getAttentionMask());
            Assert.assertEquals(specialTokenMask, encoding.getSpecialTokenMask());

            CharSpan[] charSpansExpected = {
                null,
                new CharSpan(0, 5),
                new CharSpan(5, 6),
                new CharSpan(7, 8),
                new CharSpan(8, 9),
                new CharSpan(9, 12),
                new CharSpan(12, 13),
                new CharSpan(14, 17),
                new CharSpan(18, 21),
                new CharSpan(22, 25),
                new CharSpan(26, 27),
                new CharSpan(28, 29),
                null
            };
            int expectedLength = charSpansExpected.length;
            CharSpan[] charSpansResult = encoding.getCharTokenSpans();

            Assert.assertEquals(expectedLength, charSpansResult.length);
            Assert.assertNull(charSpansResult[0]);
            Assert.assertNull(charSpansResult[expectedLength - 1]);

            for (int i = 1; i < expectedLength - 1; i++) {
                Assert.assertEquals(charSpansExpected[i].getStart(), charSpansResult[i].getStart());
                Assert.assertEquals(charSpansExpected[i].getEnd(), charSpansResult[i].getEnd());
            }

            Assert.assertThrows(() -> tokenizer.encode((String) null));
            Assert.assertThrows(() -> tokenizer.encode(new String[] {null}));
            Assert.assertThrows(() -> tokenizer.encode(null, null));
            Assert.assertThrows(() -> tokenizer.encode("null", null));
            Assert.assertThrows(() -> tokenizer.batchEncode(new String[] {null}));
            List<String> empty = Collections.singletonList(null);
            List<String> some = Collections.singletonList("null");

            Assert.assertThrows(() -> tokenizer.batchEncode(new PairList<>(empty, some)));
            Assert.assertThrows(() -> tokenizer.batchEncode(new PairList<>(some, empty)));
        }

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("addSpecialTokens", "false");
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            String[] exp = {"Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"};

            Encoding encoding = tokenizer.encode(inputs);
            Assert.assertEquals(encoding.getTokens(), exp);

            encoding = tokenizer.encode(Arrays.asList(inputs));
            Assert.assertEquals(encoding.getTokens(), exp);
        }

        DownloadUtils.download(
                "https://resources.djl.ai/test-models/tokenizer/bert-base-cased/tokenizer.json",
                "build/tokenizer/tokenizer.json");
        Path path = Paths.get("build/tokenizer");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(path)) {
            long[] ids = {101, 8667, 102, 1731, 1132, 1128, 102};
            Encoding encoding = tokenizer.encode("Hello", "How are you");
            Assert.assertEquals(encoding.getIds(), ids);

            PairList<String, String> batch = new PairList<>(2);
            batch.add("Hello", "How are you");
            batch.add("Hi, you all", "I'm fine.");
            Encoding[] encodings = tokenizer.batchEncode(batch);
            Assert.assertEquals(encodings.length, 2);
            Assert.assertEquals(encodings[0].getIds().length, 12);
        }

        Assert.assertThrows(
                () -> {
                    Path file = Paths.get("build/tokenizer/non-exists.json");
                    HuggingFaceTokenizer.builder().optTokenizerPath(file).build();
                });
    }

    @Test
    public void testDoLowerCase() throws IOException {
        String input = "Hello, y'all! How are you 😁 ?";
        String[] inputs = {"Hello, y'all!", "How are you 😁 ?"};
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optAddSpecialTokens(false)
                        .optDoLowerCase(true)
                        .build()) {
            Encoding encoding = tokenizer.encode(inputs);
            String sentence = tokenizer.buildSentence(Arrays.asList(encoding.getTokens()));
            Assert.assertEquals(sentence, "hello , y ' all ! how are you [UNK] ?");

            encoding = tokenizer.encode(input);
            Assert.assertEquals(encoding.getTokens().length, 11);

            encoding = tokenizer.encode(input, "How are you my friend");
            Assert.assertEquals(encoding.getTokens().length, 16);

            Encoding[] encodings = tokenizer.batchEncode(inputs);
            Assert.assertEquals(encodings.length, 2);

            PairList<String, String> batch = new PairList<>(2);
            batch.add("Hello", "How are you");
            batch.add("Hi, you all", "I'm fine.");
            encodings = tokenizer.batchEncode(batch);
            Assert.assertEquals(encodings.length, 2);
        }
    }

    @Test
    public void testTokenizerDecoding() throws IOException {
        long[][] testIds = {
            {101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 100, 136, 102},
            {101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 146, 112, 173, 1474, 102}
        };
        String[] expectedDecodedNoSpecialTokens = {
            "Hello, y ' all! How are you?", "Today is a sunny day. Good weather I ' d say"
        };
        String[] expectedDecodedWithSpecialTokens = {
            "[CLS] Hello, y ' all! How are you [UNK]? [SEP]",
            "[CLS] Today is a sunny day. Good weather I ' d say [SEP]"
        };
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            for (int i = 0; i < testIds.length; ++i) {
                String ret = tokenizer.decode(testIds[i]);
                Assert.assertEquals(ret, expectedDecodedWithSpecialTokens[i]);
                ret = tokenizer.decode(testIds[i], true);
                Assert.assertEquals(ret, expectedDecodedNoSpecialTokens[i]);
            }
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optAddSpecialTokens(false)
                        .build()) {
            for (int i = 0; i < testIds.length; ++i) {
                String ret = tokenizer.decode(testIds[i]);
                Assert.assertEquals(ret, expectedDecodedNoSpecialTokens[i]);
                ret = tokenizer.decode(testIds[i], false);
                Assert.assertEquals(ret, expectedDecodedWithSpecialTokens[i]);
            }
        }
    }

    @Test
    public void testMaxLengthTruncationAndAllPaddings() throws IOException {
        String[] inputs = {
            "Hello, y'all! How are you?", "Today is a sunny day. Good weather I'd say", "I am happy"
        };

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("tokenizer", "bert-base-cased");
        options.put("truncation", "longest_first"); // true
        options.put("padding", "longest"); // true
        options.put("maxLength", "10");

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.builder(options).build()) {
            int expectedSize = 10;
            for (Encoding encoding : tokenizer.batchEncode(inputs)) {
                Assert.assertEquals(encoding.getIds().length, expectedSize);
            }
        }

        options.put("padToMultipleOf", "3");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.builder(options).build()) {
            int expectedSize = 12;
            for (Encoding encoding : tokenizer.batchEncode(inputs)) {
                Assert.assertEquals(encoding.getIds().length, expectedSize);
            }
        }

        options.put("padding", "false");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.builder(options).build()) {
            int[] expectedIdsNoPadding = {10, 10, 5};
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(encodings[i].getIds().length, expectedIdsNoPadding[i]);
            }
        }
    }

    @Test
    public void testMaxModelLengthTruncationAndAllPaddings() throws IOException {
        String repeat = "hi,";
        int numRepeats = 513;
        int capacity = numRepeats * 3;
        StringBuilder stringBuilder = new StringBuilder(capacity);
        for (int i = 0; i < numRepeats; ++i) {
            stringBuilder.append(repeat);
        }
        List<String> inputs = Arrays.asList(stringBuilder.toString(), "This is a short sentence");
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("tokenizer", "bert-base-cased");
        options.put("truncation", "false");
        options.put("padding", "false");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.builder(options).build()) {
            int[] expectedNumberOfIdsNoTruncationNoPadding = new int[] {numRepeats * 2 + 2, 7};
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds().length, expectedNumberOfIdsNoTruncationNoPadding[i]);
            }
        }

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (Encoding encoding : encodings) {
                Assert.assertEquals(encoding.getIds().length, 512);
            }
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optPadding(false)
                        .build()) {
            int[] expectedSize = new int[] {512, 7};
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(encodings[i].getIds().length, expectedSize[i]);
            }
        }
    }

    @Test
    public void testNoTruncationAndAllPaddings() throws IOException {
        String[] inputs = {
            "Hello, y'all! How are you?", "Today is a sunny day. Good weather I'd say", "I am happy"
        };
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            int expectedSize = 14;
            for (Encoding encoding : tokenizer.batchEncode(inputs)) {
                Assert.assertEquals(encoding.getIds().length, expectedSize);
            }
        }

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("tokenizer", "bert-base-cased");
        options.put("padding", "false");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.builder(options).build()) {
            int[] expectedSize = {12, 14, 5};
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(encodings[i].getIds().length, expectedSize[i]);
            }
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optPadToMaxLength()
                        .build()) {
            for (Encoding encoding : tokenizer.batchEncode(inputs)) {
                Assert.assertEquals(encoding.getIds().length, 512);
            }
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optPadToMaxLength()
                        .optMaxLength(15)
                        .build()) {
            int expectedSize = 15;
            for (Encoding encoding : tokenizer.batchEncode(inputs)) {
                Assert.assertEquals(encoding.getIds().length, expectedSize);
            }
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optPadToMaxLength()
                        .optMaxLength(15)
                        .optPadToMultipleOf(8)
                        .build()) {
            int expectedSize = 16;
            for (Encoding encoding : tokenizer.batchEncode(inputs)) {
                Assert.assertEquals(encoding.getIds().length, expectedSize);
            }
        }
    }

    @Test
    public void testTruncationStride() throws IOException {
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optAddSpecialTokens(false)
                        .optWithOverflowingTokens(true)
                        .optTruncation(true)
                        .optPadding(false)
                        .optMaxLength(3)
                        .optStride(1)
                        .build()) {
            String[] inputs = {"Hello there my good friend", "How are you today"};
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            int[] expectedOverflowEncodings = {1, 1};
            int[][] expectedNumberOfOverflowingTokens = {{3}, {2}};
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getOverflowing().length, expectedOverflowEncodings[i]);
                for (int j = 0; j < expectedOverflowEncodings[i]; ++j) {
                    Assert.assertEquals(
                            encodings[i].getOverflowing()[j].getTokens().length,
                            expectedNumberOfOverflowingTokens[i][j]);
                }
            }
        }
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optAddSpecialTokens(false)
                        .optWithOverflowingTokens(true)
                        .optTruncation(true)
                        .optPadding(false)
                        .optMaxLength(8)
                        .optStride(2)
                        .build()) {
            String text = "Hello there my friend I am happy to see you";
            String textPair = "How are you my friend";
            Encoding encoding = tokenizer.encode(text, textPair);
            Assert.assertTrue(encoding.exceedMaxLength());
            Encoding[] overflowing = encoding.getOverflowing();

            int expectedNumberOfOverflowEncodings = 7;
            Assert.assertEquals(overflowing.length, expectedNumberOfOverflowEncodings);
            int[] expectedOverflowEncodingLengths = {8, 7, 8, 7, 8, 7, 7};
            for (int i = 0; i < expectedNumberOfOverflowEncodings; ++i) {
                Assert.assertEquals(
                        overflowing[i].getIds().length, expectedOverflowEncodingLengths[i]);
            }
        }
    }

    @Test
    public void testTruncationAndPaddingForPairInputs() throws IOException {
        String text = "Hello there my friend";
        String textPair = "How are you";

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optTruncateFirstOnly()
                        .optPadding(true)
                        .optMaxLength(8)
                        .build()) {
            Encoding encoding = tokenizer.encode(text, textPair);
            Assert.assertEquals(encoding.getIds().length, 8);
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optTruncateFirstOnly()
                        .optPadding(true)
                        .optMaxLength(8)
                        .optPadToMultipleOf(3)
                        .build()) {
            Encoding encoding = tokenizer.encode(text, textPair);
            Assert.assertEquals(encoding.getIds().length, 9);
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optTruncateSecondOnly()
                        .optMaxLength(8)
                        .optDoLowerCase(Locale.ROOT.toLanguageTag())
                        .build()) {
            Encoding encoding = tokenizer.encode(text, textPair);
            Assert.assertEquals(encoding.getIds().length, 8);
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optTruncateSecondOnly()
                        .optMaxLength(8)
                        .optPadToMultipleOf(3)
                        .optPadding(true)
                        .build()) {
            Encoding encoding = tokenizer.encode(text, textPair);
            Assert.assertEquals(encoding.getIds().length, 9);
        }

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optTruncation(true)
                        .optPadding(true)
                        .optMaxLength(8)
                        .build()) {
            Encoding encoding = tokenizer.encode(text, textPair);
            Assert.assertEquals(encoding.getIds().length, 8);
        }
    }

    @Test
    public void testSpecialTokenHandling() throws IOException {
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("distilbert-base-uncased")
                        .build()) {
            String someText = "¥$9";
            Encoding encodedText = tokenizer.encode(someText);

            CharSpan[] expected =
                    new CharSpan[] {
                        new CharSpan(-1, -1),
                        new CharSpan(0, 1),
                        new CharSpan(1, 2),
                        new CharSpan(2, 3),
                        new CharSpan(-1, -1)
                    };
            CharSpan[] charSpans = encodedText.getCharTokenSpans();
            for (int i = 1; i < charSpans.length - 1; i++) {
                Assert.assertEquals(expected[i].getStart(), charSpans[i].getStart());
                Assert.assertEquals(expected[i].getEnd(), charSpans[i].getEnd());
            }
        }
    }

    @Test
    public void testBatchProcessing() throws IOException {
        String[] inputs =
                new String[] {
                    "Hello there friend", "How are you today", "Good weather I'd say", "I am Happy!"
                };
        String[] outputsWithSpecialTokens =
                new String[] {
                    "[CLS] Hello there friend [SEP]",
                    "[CLS] How are you today [SEP]",
                    "[CLS] Good weather I ' d say [SEP]",
                    "[CLS] I am Happy! [SEP]"
                };
        String[] outputsWithoutSpecialTokens =
                new String[] {
                    "Hello there friend",
                    "How are you today",
                    "Good weather I ' d say",
                    "I am Happy!"
                };
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optPadding(false)
                        .build()) {

            // default tokenizer with special tokens included
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            long[][] batchIds =
                    Arrays.stream(encodings).map(Encoding::getIds).toArray(long[][]::new);
            String[] outputs = tokenizer.batchDecode(batchIds);
            Assert.assertEquals(outputs, outputsWithSpecialTokens);

            // encode with special tokens, decode with special tokens
            encodings = tokenizer.batchEncode(inputs, true, false);
            batchIds = Arrays.stream(encodings).map(Encoding::getIds).toArray(long[][]::new);
            outputs = tokenizer.batchDecode(batchIds, false);
            Assert.assertEquals(outputs, outputsWithSpecialTokens);

            // encode without special tokens, decode without special tokens
            encodings = tokenizer.batchEncode(inputs, false, false);
            batchIds = Arrays.stream(encodings).map(Encoding::getIds).toArray(long[][]::new);
            outputs = tokenizer.batchDecode(batchIds, true);
            Assert.assertEquals(outputs, outputsWithoutSpecialTokens);
        }
    }

    @Test
    public void testTokenizerWithPresetPaddingConfiguration() throws IOException {
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerPath(
                                Paths.get("src/test/resources/fake-tokenizer-with-padding/"))
                        .optMaxLength(8)
                        .optPadToMaxLength()
                        .build()) {
            Encoding encoding = tokenizer.encode("test sentence");
            String[] tokens = encoding.getTokens();
            String[] expected = {
                "<s>", "▁", "test", "▁sentence", "</s>", "<pad>", "<pad>", "<pad>"
            };
            Assert.assertEquals(tokens, expected);
        }
    }

    @Test
    public void testAuthToken() {
        TestRequirements.notOffline();
        System.setProperty("HF_TOKEN", "test_token");
        try {
            HuggingFaceTokenizer.newInstance("mistralai/Mixtral-8x7B-v0.1");
        } catch (RuntimeException ignore) {
            // access denied
        } finally {
            System.clearProperty("HF_TOKEN");
        }
    }
}
