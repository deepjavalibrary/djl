/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext;

import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class FtModelTest {

    @Test
    public void testModelNameOutsideModelDir() throws IOException {
        if (Utils.isFileOutsideModelDirAllowed()) {
            throw new SkipException("Files outside the model directory are allowed");
        }
        Path root = Files.createTempDirectory("djl-ft-model-name");
        try (FtModel model = new FtModel("model")) {
            Path modelDir = Files.createDirectories(root.resolve("model"));
            // The file "../outside" would select if the name were not contained.
            Files.write(root.resolve("outside.ftz"), new byte[] {0});
            IllegalArgumentException e =
                    Assert.expectThrows(
                            IllegalArgumentException.class,
                            () -> model.load(modelDir, "../outside"));
            Assert.assertTrue(
                    e.getMessage().contains("outside the model directory"), e.getMessage());
        } finally {
            Utils.deleteQuietly(root);
        }
    }
}
