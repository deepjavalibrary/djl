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

public class LocalTest {

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
    public void testLocal() throws IOException {
        String[] content = {"Hello, this is model", "Hello, this is synset"};
        TestUtils.writeTmpFile(modelDir, "resnet101.pt", content[0]);
        TestUtils.writeTmpFile(modelDir, "synset.txt.gz", content[1]);
        TestUtils.writeTmpFile(modelDir.resolve("inside"), "synset.txt", content[1]);
        String description = "This is a test image clasisification";
        String name = "Image Classification";
        String groupId = "ai.djl.pytorch";
        String artifactId = "resnet";
        String artifactVersion = "0.0.3";
        String artifactName = "traced_resnet101";
        MetadataBuilder.builder()
                .setBaseDir(destDir.toAbsolutePath().toString())
                .setArtifactDir(modelDir.toAbsolutePath().toString())
                .setName(name)
                .setDescription(description)
                .setApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setGroupId(groupId)
                .setArtifactName(artifactName)
                .setArtifactId(artifactId)
                .addProperty("hey", "10")
                .addArgument("hey", "value")
                .optArtifactVersion(artifactVersion)
                .buildLocal();
        Path targetPath =
                destDir.resolve(
                        Paths.get(
                                "mlrepo/model",
                                Application.CV.IMAGE_CLASSIFICATION.getPath(),
                                groupId.replace(".", "/"),
                                artifactId));
        Assert.assertTrue(targetPath.resolve("metadata.json").toFile().exists());
        Assert.assertTrue(
                targetPath
                        .resolve(artifactVersion + "/" + artifactName + ".pt.gz")
                        .toFile()
                        .exists());
        Assert.assertTrue(
                targetPath.resolve(artifactVersion + "/" + "synset.txt.gz").toFile().exists());
        Assert.assertTrue(
                targetPath.resolve(artifactVersion + "/" + "inside.zip").toFile().exists());
        Metadata metadata = TestUtils.readGson(targetPath.resolve("metadata.json"));
        Assert.assertEquals(metadata.getName(), name);
        Assert.assertEquals(metadata.getDescription(), description);
        Assert.assertEquals(metadata.getGroupId(), groupId);
        Assert.assertEquals(metadata.getArtifactId(), artifactId);
        Artifact artifact = metadata.getArtifacts().get(0);
        Assert.assertEquals(artifact.getName(), artifactName);
        Assert.assertEquals(artifact.getProperties().get("hey"), "10");
        Assert.assertTrue(artifact.getFiles().containsKey("model"));
        Assert.assertTrue(artifact.getFiles().containsKey("synset"));
        Assert.assertTrue(artifact.getFiles().containsKey("inside"));
    }

    @Test
    public void testDataset() throws IOException {
        String[] content = {"Hello, this is dataset", "Hello, this is synset"};
        TestUtils.writeTmpFile(modelDir, "dataset.csv", content[0]);
        TestUtils.writeTmpFile(modelDir, "synset.txt", content[1]);
        String description = "This is a test text dataset";
        String name = "Rich dataset";
        String groupId = "ai.djl.basicdataset";
        String artifactId = "tatoeba_en_cn";
        String artifactVersion = "1.0";
        String artifactName = "tatoeba_en_cn";
        MetadataBuilder.builder()
                .setBaseDir(destDir.toAbsolutePath().toString())
                .setArtifactDir(modelDir.toAbsolutePath().toString())
                .setName(name)
                .setDescription(description)
                .setApplication(Application.NLP.TEXT_CLASSIFICATION)
                .setGroupId(groupId)
                .setArtifactName(artifactName)
                .setArtifactId(artifactId)
                .optArtifactVersion(artifactVersion)
                .optIsDataset()
                .buildLocal();
        Path targetPath =
                destDir.resolve(
                        Paths.get(
                                "mlrepo/dataset",
                                Application.NLP.TEXT_CLASSIFICATION.getPath(),
                                groupId.replace(".", "/"),
                                artifactId));
        Assert.assertTrue(targetPath.resolve("metadata.json").toFile().exists());
        Assert.assertTrue(
                targetPath.resolve(artifactVersion + "/" + "synset.txt.gz").toFile().exists());
        Assert.assertTrue(
                targetPath.resolve(artifactVersion + "/" + "dataset.csv.gz").toFile().exists());
    }

    @Test
    public void testMerge() throws IOException {
        String[] content = {"Hello, this is model", "Hello, this is synset"};
        TestUtils.writeTmpFile(modelDir, "resnet101.pt", content[0]);
        TestUtils.writeTmpFile(modelDir, "synset.txt.gz", content[1]);
        TestUtils.writeTmpFile(modelDir.resolve("inside"), "synset.txt", content[1]);
        String description = "This is a test image clasisification";
        String name = "Image Classification";
        String groupId = "ai.djl.pytorch";
        String artifactId = "resnet";
        String artifactVersion = "0.0.3";
        String artifactName1 = "traced_resnet101";
        MetadataBuilder.builder()
                .setBaseDir(destDir.toAbsolutePath().toString())
                .setArtifactDir(modelDir.toAbsolutePath().toString())
                .setName(name)
                .setDescription(description)
                .setApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setGroupId(groupId)
                .setArtifactName(artifactName1)
                .setArtifactId(artifactId)
                .addProperty("hey", "10")
                .addArgument("hey", "value")
                .optArtifactVersion(artifactVersion)
                .buildLocal();
        // prepare the merge operation
        FileUtils.deleteDirectory(modelDir.toFile());
        TestUtils.writeTmpFile(modelDir, "resnet50.pt", content[0]);
        TestUtils.writeTmpFile(modelDir, "synset.txt", content[1]);
        Path targetPath =
                destDir.resolve(
                        Paths.get(
                                "mlrepo/model",
                                Application.CV.IMAGE_CLASSIFICATION.getPath(),
                                groupId.replace(".", "/"),
                                artifactId));
        String artifactName2 = "traced_resnet50";
        MetadataBuilder.builder()
                .setBaseDir(destDir.toAbsolutePath().toString())
                .setArtifactDir(modelDir.toAbsolutePath().toString())
                .setName(name)
                .setDescription(description)
                .setApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setGroupId(groupId)
                .setArtifactName(artifactName2)
                .setArtifactId(artifactId)
                .addProperty("hey", "50")
                .addArgument("hey", "value")
                .optArtifactVersion(artifactVersion)
                .buildLocal();
        Assert.assertTrue(targetPath.resolve("metadata.json").toFile().exists());
        Assert.assertTrue(
                targetPath
                        .resolve(artifactVersion + "/" + artifactName1 + ".pt.gz")
                        .toFile()
                        .exists());
        Assert.assertTrue(
                targetPath.resolve("0.0.4" + "/" + artifactName2 + ".pt.gz").toFile().exists());
        Metadata metadata = TestUtils.readGson(targetPath.resolve("metadata.json"));
        List<Artifact> artifacts = metadata.getArtifacts();
        Assert.assertEquals(artifacts.size(), 2);
        Assert.assertEquals(artifacts.get(0).getName(), artifactName1);
        Assert.assertEquals(artifacts.get(1).getName(), artifactName2);
        Assert.assertEquals(artifacts.get(0).getVersion(), "0.0.3");
        Assert.assertEquals(artifacts.get(1).getVersion(), "0.0.4");
    }

    @AfterMethod
    @AfterTest
    public void removeFiles() {
        FileUtils.deleteQuietly(modelDir.toFile());
        FileUtils.deleteQuietly(destDir.toFile());
    }
}
