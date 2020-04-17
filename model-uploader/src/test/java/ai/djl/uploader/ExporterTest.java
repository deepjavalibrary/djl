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

package ai.djl.uploader;

import ai.djl.Application;
import ai.djl.uploader.util.TestUtils;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ExporterTest {
    private final String baseDir = System.getProperty("java.io.tmpdir");
    private Path modelDir;
    private Path destDir;

    @BeforeTest
    public void setUp() {
        modelDir = Paths.get(baseDir, "uploader/model");
        destDir = Paths.get(baseDir, "uploader/dest");
        FileUtils.deleteQuietly(modelDir.toFile());
        FileUtils.deleteQuietly(destDir.toFile());
    }

    @Test
    public void testModel() throws IOException, InterruptedException {
        String[] content = {"Hello, this is model", "Hello, this is synset"};
        TestUtils.writeTmpFile(modelDir, "resnet101.pt", content[0]);
        TestUtils.writeTmpFile(modelDir, "synset.txt.gz", content[1]);
        String artifactName = "resnet101";
        String artifactId = "resnet";
        String inputDir = modelDir.toAbsolutePath().toString();
        String destPath = destDir.toAbsolutePath().toString();
        String[] args = {
            "-model",
            "-an",
            artifactName,
            "-ai",
            artifactId,
            "-b",
            inputDir,
            "-ap",
            "image classification",
            "-o",
            destPath
        };
        Exporter.main(args);
        Path targetPath =
                destDir.resolve(
                        Paths.get(
                                "mlrepo/model",
                                Application.CV.IMAGE_CLASSIFICATION.getPath(),
                                "ai.djl.model".replace(".", "/"),
                                artifactId));
        Assert.assertTrue(targetPath.resolve("metadata.json").toFile().exists());
        Assert.assertTrue(targetPath.resolve("0.0.1/" + artifactName + ".pt.gz").toFile().exists());
        Assert.assertTrue(targetPath.resolve("0.0.1/" + "synset.txt.gz").toFile().exists());
    }

    @AfterMethod
    @AfterTest
    public void removeFiles() {
        FileUtils.deleteQuietly(modelDir.toFile());
        FileUtils.deleteQuietly(destDir.toFile());
    }
}
