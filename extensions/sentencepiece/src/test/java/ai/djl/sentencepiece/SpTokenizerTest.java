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

package ai.djl.sentencepiece;

import ai.djl.testing.TestRequirements;
import ai.djl.training.util.DownloadUtils;

import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class SpTokenizerTest {

    @BeforeTest
    public void downloadModel() throws IOException {
        TestRequirements.notArm();

        Path modelFile = Paths.get("build/test/sp_model/sp_model.model");
        if (Files.notExists(modelFile)) {
            DownloadUtils.download(
                    "https://resources.djl.ai/test-models/sententpiece_test_model.model",
                    "build/test/sp_model/sp_model.model");
        }
    }

    @Test
    public void testLoadFromBytes() throws IOException {
        Path modelPath = Paths.get("build/test/sp_model/sp_model.model");
        byte[] bytes = Files.readAllBytes(modelPath);
        try (SpTokenizer tokenizer = new SpTokenizer(bytes)) {
            String original = "Hello World";
            List<String> tokens = tokenizer.tokenize(original);
            List<String> expected = Arrays.asList("▁He", "ll", "o", "▁", "W", "or", "l", "d");
            Assert.assertEquals(tokens, expected);
            String recovered = tokenizer.buildSentence(tokens);
            Assert.assertEquals(original, recovered);
        }
    }

    @Test
    public void testLoadFromInputStream() throws IOException {
        Path modelPath = Paths.get("build/test/sp_model/sp_model.model");
        try (InputStream is = Files.newInputStream(modelPath)) {
            try (SpTokenizer tokenizer = new SpTokenizer(is)) {
                String original = "Hello World";
                List<String> tokens = tokenizer.tokenize(original);
                List<String> expected = Arrays.asList("▁He", "ll", "o", "▁", "W", "or", "l", "d");
                Assert.assertEquals(tokens, expected);
                String recovered = tokenizer.buildSentence(tokens);
                Assert.assertEquals(original, recovered);
            }
        }
    }

    @Test
    public void testTokenize() throws IOException {
        TestRequirements.notWindows();

        Path modelPath = Paths.get("build/test/sp_model");
        try (SpTokenizer tokenizer = new SpTokenizer(modelPath)) {
            String original = "Hello World";
            List<String> tokens = tokenizer.tokenize(original);
            List<String> expected = Arrays.asList("▁He", "ll", "o", "▁", "W", "or", "l", "d");
            Assert.assertEquals(tokens, expected);
            String recovered = tokenizer.buildSentence(tokens);
            Assert.assertEquals(original, recovered);
        }
    }

    @Test
    @SuppressWarnings("AvoidEscapedUnicodeCharacters")
    public void testUtf16Tokenize() throws IOException {
        TestRequirements.notWindows();

        Path modelPath = Paths.get("build/test/sp_model/sp_model.model");
        try (SpTokenizer tokenizer = new SpTokenizer(modelPath)) {
            String original = "\uD83D\uDC4B\uD83D\uDC4B";
            List<String> tokens = tokenizer.tokenize(original);
            List<String> expected = Arrays.asList("▁", "\uD83D\uDC4B\uD83D\uDC4B");
            Assert.assertEquals(tokens, expected);
        }
    }

    @Test
    public void testEncodeDecode() throws IOException {
        TestRequirements.notWindows();

        Path modelPath = Paths.get("build/test/sp_model");
        String prefix = "sp_model";
        try (SpTokenizer tokenizer = new SpTokenizer(modelPath, prefix)) {
            String original = "Hello World";
            SpProcessor processor = tokenizer.getProcessor();
            int[] ids = processor.encode(original);
            int[] expected = new int[] {151, 88, 21, 4, 321, 54, 31, 17};
            Assert.assertEquals(ids, expected);
            String recovered = processor.decode(ids);
            Assert.assertEquals(recovered, original);
        }
    }

    @Test
    public void testModelNotFound() throws IOException {
        TestRequirements.notWindows();

        Assert.assertThrows(
                () -> {
                    new SpTokenizer(Paths.get("build/test/non-exists"));
                });

        Assert.assertThrows(
                () -> {
                    new SpTokenizer(Paths.get("build/test/sp_model"), "non-exists.model");
                });

        Assert.assertThrows(
                () -> {
                    new SpTokenizer(Paths.get("build/test/sp_model"), "non-exists");
                });
    }
}
