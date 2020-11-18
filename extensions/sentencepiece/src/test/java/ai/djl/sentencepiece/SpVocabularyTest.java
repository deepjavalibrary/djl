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
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class SpVocabularyTest {

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
    public void testTokenIdConversion() throws IOException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Skip windows test.");
        }
        Path modelPath = Paths.get("build/test/models");
        String prefix = "sententpiece_test_model";
        SpTokenizer tokenizer = new SpTokenizer(modelPath, prefix);
        SpVocabulary vocabulary = SpVocabulary.from(tokenizer);
        String expectedToken = "<s>";
        Assert.assertEquals(vocabulary.getToken(1), expectedToken);
        Assert.assertEquals(vocabulary.getIndex("l"), 31);
    }
}
