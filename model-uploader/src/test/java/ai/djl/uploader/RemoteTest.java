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
import ai.djl.repository.Artifact;
import ai.djl.repository.Metadata;
import ai.djl.uploader.util.TestUtils;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class RemoteTest {

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
    public void testRemoteMerge() throws IOException {
        String[] content = {"Hello, this is model", "Hello, this is synset"};
        TestUtils.writeTmpFile(modelDir, "resnet1024_v1d-symbol.json", content[0]);
        TestUtils.writeTmpFile(modelDir, "resnet1024_v1d-0000.params", content[0]);
        TestUtils.writeTmpFile(modelDir, "synset.txt", content[1]);
        String description = "This is a test image clasisification";
        String name = "Image Classification";
        String groupId = "ai.djl.mxnet";
        String artifactId = "resnet";
        String artifactVersion = "0.0.1";
        String artifactName = "resnet1024";
        MetadataBuilder.builder()
                .setBaseDir(destDir.toAbsolutePath().toString())
                .setArtifactDir(modelDir)
                .setName(name)
                .setDescription(description)
                .setApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setGroupId(groupId)
                .setArtifactName(artifactName)
                .setArtifactId(artifactId)
                .addProperty("layers", "1024")
                .addProperty("flavor", "v1")
                .addProperty("dataset", "imagenet")
                .addArgument("width", 224)
                .addArgument("height", 224)
                .optArtifactVersion(artifactVersion)
                .buildExternal();
        Path targetPath =
                destDir.resolve(
                        Paths.get(
                                "mlrepo/model",
                                Application.CV.IMAGE_CLASSIFICATION.getPath(),
                                groupId.replace(".", "/"),
                                artifactId));
        Assert.assertTrue(targetPath.resolve("metadata.json").toFile().exists());
        Assert.assertTrue(
                targetPath.resolve(artifactVersion + "/" + "synset.txt.gz").toFile().exists());
        Assert.assertTrue(
                targetPath
                        .resolve(artifactVersion + "/" + "resnet1024-0000.params.gz")
                        .toFile()
                        .exists());
        Metadata metadata = TestUtils.readGson(targetPath.resolve("metadata.json"));
        List<Artifact> artifacts = metadata.getArtifacts();
        Artifact artifact = artifacts.get(artifacts.size() - 1);
        Assert.assertEquals(artifact.getName(), artifactName);
        Assert.assertEquals(artifact.getProperties().get("layers"), "1024");
    }

    @AfterMethod
    @AfterTest
    public void removeFiles() {
        FileUtils.deleteQuietly(modelDir.toFile());
        FileUtils.deleteQuietly(destDir.toFile());
    }
}
