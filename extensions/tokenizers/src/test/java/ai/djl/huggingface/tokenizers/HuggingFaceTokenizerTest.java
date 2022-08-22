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

import ai.djl.huggingface.tokenizers.jni.CharSpan;
import ai.djl.training.util.DownloadUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class HuggingFaceTokenizerTest {

    @Test
    public void testTokenizer() throws IOException {
        String input = "Hello, y'all! How are you 😁 ?";
        String[] inputs = {"Hello, y'all!", "How are you 😁 ?"};

        String[] expected = {
            "[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"
        };

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            List<String> ret = tokenizer.tokenize(input);
            Assert.assertEquals(ret.toArray(new String[0]), expected);
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
                new CharSpan(26, 30),
                new CharSpan(31, 32),
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

            Encoding[] encodings = tokenizer.batchEncode(Arrays.asList(inputs));
            Assert.assertEquals(encodings.length, 2);
        }

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("addSpecialTokens", "false");
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            String[] exp = {"Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"};

            Encoding encoding = tokenizer.encode(Arrays.asList(inputs));
            Assert.assertEquals(encoding.getTokens(), exp);
        }

        DownloadUtils.download(
                "https://huggingface.co/bert-base-cased/raw/main/tokenizer.json",
                "build/tokenizer/tokenizer.json");
        Path path = Paths.get("build/tokenizer");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(path)) {
            long[] ids = {101, 8667, 102, 1731, 1132, 1128, 102};
            Encoding encoding = tokenizer.encode("Hello", "How are you");
            Assert.assertEquals(encoding.getIds(), ids);
        }
    }

    @Test
    public void testTokenizerDecoding() {
        List<long[]> testIds =
                Arrays.asList(
                        new long[] {
                            101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 100, 136, 102
                        },
                        new long[] {
                            101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 146, 112, 173, 1474,
                            102
                        });
        List<String> expectedDecodingsNoSpecialTokens =
                Arrays.asList(
                        "Hello, y ' all! How are you?",
                        "Today is a sunny day. Good weather I ' d say");
        List<String> expectedDecodingsWithSpecialTokens =
                Arrays.asList(
                        "[CLS] Hello, y ' all! How are you [UNK]? [SEP]",
                        "[CLS] Today is a sunny day. Good weather I ' d say [SEP]");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            for (int i = 0; i < testIds.size(); ++i) {
                Assert.assertEquals(
                        tokenizer.decode(testIds.get(i)),
                        expectedDecodingsWithSpecialTokens.get(i));
                Assert.assertEquals(
                        tokenizer.decode(testIds.get(i), true),
                        expectedDecodingsNoSpecialTokens.get(i));
            }
        }

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("addSpecialTokens", "false");
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            for (int i = 0; i < testIds.size(); ++i) {
                Assert.assertEquals(
                        tokenizer.decode(testIds.get(i)), expectedDecodingsNoSpecialTokens.get(i));
                Assert.assertEquals(
                        tokenizer.decode(testIds.get(i), false),
                        expectedDecodingsWithSpecialTokens.get(i));
            }
        }
    }

    @Test
    public void testMaxLengthTruncationAndAllPaddings() {
        List<String> inputs =
                Arrays.asList(
                        "Hello, y'all! How are you?",
                        "Today is a sunny day. Good weather I'd say",
                        "I am happy");

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("truncation", "longest_first");
        options.put("padding", "longest");
        options.put("maxLength", "10");

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            List<long[]> expectedIdsWithTruncationAndPadding =
                    Arrays.asList(
                            new long[] {101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 102},
                            new long[] {101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 102},
                            new long[] {101, 146, 1821, 2816, 102, 0, 0, 0, 0, 0});
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds(), expectedIdsWithTruncationAndPadding.get(i));
            }

            List<long[]> expectedIdsWithTruncationAndPaddingToMultiple =
                    Arrays.asList(
                            new long[] {101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 102, 0, 0},
                            new long[] {
                                101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 102, 0, 0
                            },
                            new long[] {101, 146, 1821, 2816, 102, 0, 0, 0, 0, 0, 0, 0});
            tokenizer.setPadToMultipleOf(3);
            encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds(),
                        expectedIdsWithTruncationAndPaddingToMultiple.get(i));
            }

            List<long[]> expectedIdsNoPadding =
                    Arrays.asList(
                            new long[] {101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 102},
                            new long[] {101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 102},
                            new long[] {101, 146, 1821, 2816, 102});
            tokenizer.setPadding(false);
            encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(encodings[i].getIds(), expectedIdsNoPadding.get(i));
            }
        }
    }

    @Test
    public void testMaxModelLengthTruncationAndAllPaddings() {
        String repeat = "hi,";
        int numRepeats = 513;
        int capacity = numRepeats * 3;
        StringBuilder stringBuilder = new StringBuilder(capacity);
        for (int i = 0; i < numRepeats; ++i) {
            stringBuilder.append(repeat);
        }
        List<String> inputs = Arrays.asList(stringBuilder.toString(), "This is a short sentence");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            int[] expectedNumberOfIdsNoTruncationNoPadding = new int[] {numRepeats * 2 + 2, 7};
            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds().length, expectedNumberOfIdsNoTruncationNoPadding[i]);
            }

            tokenizer.setTruncation(true);
            int[] expectedNumberOfIdsDefaultTruncationNoPadding = new int[] {512, 7};
            encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds().length,
                        expectedNumberOfIdsDefaultTruncationNoPadding[i]);
            }

            tokenizer.setPadding(true);
            encodings = tokenizer.batchEncode(inputs);
            for (Encoding encoding : encodings) {
                Assert.assertEquals(encoding.getIds().length, 512);
            }
        }
    }

    @Test
    public void testNoTruncationAndAllPaddings() {
        List<String> inputs =
                Arrays.asList(
                        "Hello, y'all! How are you?",
                        "Today is a sunny day. Good weather I'd say",
                        "I am happy");
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            List<long[]> expectedIdsNoTruncationOrPadding =
                    Arrays.asList(
                            new long[] {
                                101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 136, 102
                            },
                            new long[] {
                                101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 146, 112, 173,
                                1474, 102
                            },
                            new long[] {101, 146, 1821, 2816, 102});

            Encoding[] encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(encodings[i].getIds(), expectedIdsNoTruncationOrPadding.get(i));
            }
            tokenizer.setPadding(true);
            List<long[]> expectedIdsNoTruncationDefaultPadding =
                    Arrays.asList(
                            new long[] {
                                101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 136, 102, 0,
                                0
                            },
                            new long[] {
                                101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 146, 112, 173,
                                1474, 102
                            },
                            new long[] {101, 146, 1821, 2816, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds(), expectedIdsNoTruncationDefaultPadding.get(i));
            }

            tokenizer.setPaddingMaxLength();
            encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(encodings[i].getIds().length, 512);
            }

            tokenizer.setMaxLength(15);
            List<long[]> expectedIdsNoTruncationMaxLengthPadding =
                    Arrays.asList(
                            new long[] {
                                101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 136, 102, 0,
                                0, 0
                            },
                            new long[] {
                                101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 146, 112, 173,
                                1474, 102, 0
                            },
                            new long[] {101, 146, 1821, 2816, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds(), expectedIdsNoTruncationMaxLengthPadding.get(i));
            }

            tokenizer.setPadToMultipleOf(8);
            List<long[]> expectedIdsNoTruncationMaxLengthPadToMultiplePadding =
                    Arrays.asList(
                            new long[] {
                                101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 136, 102, 0,
                                0, 0, 0
                            },
                            new long[] {
                                101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 146, 112, 173,
                                1474, 102, 0, 0
                            },
                            new long[] {
                                101, 146, 1821, 2816, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            });
            encodings = tokenizer.batchEncode(inputs);
            for (int i = 0; i < encodings.length; ++i) {
                Assert.assertEquals(
                        encodings[i].getIds(),
                        expectedIdsNoTruncationMaxLengthPadToMultiplePadding.get(i));
            }
        }
    }

    @Test
    public void testTruncationAndPaddingForPairInputs() {
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("truncation", "only_first");
        options.put("padding", "longest");
        options.put("maxLength", "8");
        String text = "Hello there my friend";
        String textPair = "How are you";

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            Encoding encoding = tokenizer.encode(text, textPair);
            long[] expectedId = new long[] {101, 8667, 1175, 102, 1731, 1132, 1128, 102};
            Assert.assertEquals(encoding.getIds(), expectedId);

            tokenizer.setPadToMultipleOf(3);
            encoding = tokenizer.encode(text, textPair);
            long[] expectedIdPadToMultiple =
                    new long[] {101, 8667, 1175, 102, 1731, 1132, 1128, 102, 0};
            Assert.assertEquals(encoding.getIds(), expectedIdPadToMultiple);
        }

        options.put("truncation", "only_second");
        options.remove("padding");
        options.put("maxLength", "8");

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            Encoding encoding = tokenizer.encode(text, textPair);
            long[] expectedId = new long[] {101, 8667, 1175, 1139, 1910, 102, 1731, 102};
            Assert.assertEquals(encoding.getIds(), expectedId);

            tokenizer.setPadToMultipleOf(3);
            tokenizer.setPadding(true);
            encoding = tokenizer.encode(text, textPair);
            long[] expectedIdPadToMultiple =
                    new long[] {101, 8667, 1175, 1139, 1910, 102, 1731, 102, 0};
            Assert.assertEquals(encoding.getIds(), expectedIdPadToMultiple);
        }

        options.put("truncation", "longest_first");
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            Encoding encoding = tokenizer.encode(text, textPair);
            long[] expectedId = new long[] {101, 8667, 1175, 1139, 102, 1731, 1132, 102};
            Assert.assertEquals(encoding.getIds(), expectedId);
        }
    }
}
