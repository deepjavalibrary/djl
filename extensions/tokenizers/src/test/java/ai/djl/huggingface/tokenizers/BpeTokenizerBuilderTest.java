/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.training.util.DownloadUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

public class BpeTokenizerBuilderTest {

    @Test
    public void testTokenizerCreation() throws IOException {
        Path bpe = Paths.get("build/BPE");
        Path vocab = bpe.resolve("vocab.json");
        Path merges = bpe.resolve("merges.txt");

        DownloadUtils.download(
                new URL("https://resources.djl.ai/test-models/tokenizer/vocab.json"), vocab);
        DownloadUtils.download(
                new URL("https://resources.djl.ai/test-models/tokenizer/merges.txt"), merges);

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder().optTokenizerPath(bpe).build()) {
            Assert.assertNotNull(tokenizer);
        }
    }
}
