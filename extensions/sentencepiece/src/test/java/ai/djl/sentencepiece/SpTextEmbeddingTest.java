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
package ai.djl.sentencepiece;

import ai.djl.training.util.DownloadUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class SpTextEmbeddingTest {
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
    public void testEmbedding() throws IOException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Skip windows test.");
        }
        Path modelPath = Paths.get("build/test/models/sententpiece_test_model.model");
        try (SpTokenizer tokenizer = new SpTokenizer(modelPath)) {
            SpTextEmbedding embedding = SpTextEmbedding.from(tokenizer);
            long[] indices =
                    embedding.preprocessTextToEmbed(Collections.singletonList("Hello World"));
            long[] expected = new long[] {151, 88, 21, 4, 321, 54, 31, 17};
            Assert.assertEquals(indices, expected);
        }
    }
}
