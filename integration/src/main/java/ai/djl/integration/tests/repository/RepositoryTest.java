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
package ai.djl.integration.tests.repository;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Metadata;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class RepositoryTest {

    @BeforeClass
    public void setUp() throws IOException {
        Path dir = Paths.get("build/models/archive/");
        Files.createDirectories(dir);
        Files.createDirectories(dir.resolve("test_dir"));
        try (Writer writer = Files.newBufferedWriter(dir.resolve("test_dir/test_file.txt"))) {
            writer.append("N/A");
        }
        Path synset = dir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synset)) {
            writer.append("cat");
        }
        Path dest = Paths.get("build/models/test_model.zip");
        ZipUtils.zip(dir, dest, false);
    }

    @AfterClass
    public void tearDown() {
        Utils.deleteQuietly(Paths.get("build/models"));
    }

    @Test
    public void testSimpleRepositoryArchive() throws IOException {
        Repository repo = Repository.newInstance("archive", "build/models/test_model.zip");
        List<MRL> resources = repo.getResources();

        Assert.assertEquals(resources.size(), 1);

        Metadata metadata = repo.locate(resources.get(0));
        Assert.assertEquals(metadata.getApplication(), Application.UNDEFINED);
        Assert.assertEquals(metadata.getGroupId(), DefaultModelZoo.GROUP_ID);
        Assert.assertEquals(metadata.getArtifactId(), "test_model");

        List<Artifact> artifacts = metadata.getArtifacts();
        Assert.assertEquals(artifacts.size(), 1);

        Artifact artifact = artifacts.get(0);
        Assert.assertEquals(artifact.getName(), "test_model");

        Map<String, Artifact.Item> files = artifact.getFiles();
        Assert.assertEquals(files.size(), 1);

        Artifact.Item item = files.get("test_model");

        repo.prepare(artifact);

        Path modelPath = repo.getResourceDirectory(artifact);
        Assert.assertTrue(Files.exists(modelPath));

        String[] list = repo.listDirectory(item, "test_dir");
        Assert.assertEquals(list[0], "test_file.txt");

        List<String> classes = Utils.readLines(repo.openStream(item, "synset.txt"));
        Assert.assertEquals(classes.size(), 1);

        Utils.deleteQuietly(modelPath);
    }

    @Test
    public void testSimpleRepositoryArchiveWithQueryString() throws IOException {
        Path modelFile = Paths.get("build/models/test_model.zip").toAbsolutePath();
        String url = modelFile.toUri().toString() + "?artifact_id=resnet&model_name=resnet18";
        Repository repo = Repository.newInstance("archive", url);
        List<MRL> resources = repo.getResources();

        Assert.assertEquals(resources.size(), 1);

        Metadata metadata = repo.locate(resources.get(0));
        Assert.assertEquals(metadata.getApplication(), Application.UNDEFINED);
        Assert.assertEquals(metadata.getGroupId(), DefaultModelZoo.GROUP_ID);
        Assert.assertEquals(metadata.getArtifactId(), "resnet");

        List<Artifact> artifacts = metadata.getArtifacts();
        Assert.assertEquals(artifacts.size(), 1);

        Artifact artifact = artifacts.get(0);
        Assert.assertEquals(artifact.getName(), "resnet18");

        Map<String, Artifact.Item> files = artifact.getFiles();
        Assert.assertEquals(files.size(), 1);
    }

    @Test
    public void testSimpleRepositoryDir() throws IOException {
        Repository repo =
                Repository.newInstance(
                        "archive", "build/models/archive?artifact_id=resnet&model_name=resnet18");
        List<MRL> resources = repo.getResources();

        Assert.assertEquals(resources.size(), 1);

        Metadata metadata = repo.locate(resources.get(0));
        Assert.assertEquals(metadata.getApplication(), Application.UNDEFINED);
        Assert.assertEquals(metadata.getGroupId(), DefaultModelZoo.GROUP_ID);
        Assert.assertEquals(metadata.getArtifactId(), "resnet");

        List<Artifact> artifacts = metadata.getArtifacts();
        Assert.assertEquals(artifacts.size(), 1);

        Artifact artifact = artifacts.get(0);
        Assert.assertEquals(artifact.getName(), "resnet18");

        Map<String, Artifact.Item> files = artifact.getFiles();
        Assert.assertEquals(files.size(), 2);
    }
}
