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

import ai.djl.training.util.DownloadUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class SpTokenizerTest {

    @BeforeTest
    public void downloadModel() throws IOException {
        Path modelFile = Paths.get("build/test/models/sententpiece_test_model.model");
        if (Files.notExists(modelFile)) {
            DownloadUtils.download(
                    "https://resources.djl.ai/test-models/sententpiece_test_model.model",
                    "build/test/models/sententpiece_test_model.model");
        }
    }

    @Test
    public void testTokenize() throws IOException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Skip windows test.");
        }
        Path modelPath = Paths.get("build/test/models/sententpiece_test_model.model");
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
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Skip windows test.");
        }
        Path modelPath = Paths.get("build/test/models/sententpiece_test_model.model");
        try (SpTokenizer tokenizer = new SpTokenizer(modelPath)) {
            String original = "\uD83D\uDC4B\uD83D\uDC4B";
            List<String> tokens = tokenizer.tokenize(original);
            List<String> expected = Arrays.asList("▁", "\uD83D\uDC4B\uD83D\uDC4B");
            Assert.assertEquals(tokens, expected);
        }
    }

    @Test
    public void testEncodeDecode() throws IOException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Skip windows test.");
        }
        Path modelPath = Paths.get("build/test/models");
        String prefix = "sententpiece_test_model";
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
}
