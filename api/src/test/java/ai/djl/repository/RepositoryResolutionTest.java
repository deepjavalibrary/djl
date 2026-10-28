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
package ai.djl.repository;

import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * Covers the resolution a legitimate artifact goes through in each repository implementation.
 *
 * <p>The containment checks reject a name that leaves the directory, so each implementation also
 * has to keep resolving the names it is supposed to.
 */
public class RepositoryResolutionTest {

    private static final Path ROOT = Paths.get("build/repo-resolution");

    @AfterClass
    public void cleanup() {
        Utils.deleteQuietly(ROOT);
    }

    @Test
    public void testSimpleRepositoryResolvesAnItemName() throws IOException {
        Path dir = ROOT.resolve("simple").toAbsolutePath();
        Files.createDirectories(dir);
        Files.write(dir.resolve("synset.txt"), "cat".getBytes(StandardCharsets.UTF_8));
        SimpleRepository repo = new SimpleRepository("simple", dir.toUri(), dir);

        Artifact.Item item = new Artifact.Item();
        item.setName("synset.txt");
        item.setArtifact(artifact(dir.toUri(), "0.0.1"));
        Assert.assertEquals(Paths.get(repo.resolvePath(item, "")), dir.resolve("synset.txt"));
    }

    @Test
    public void testRemoteRepositoryResolvesIntoTheCache() throws IOException {
        RemoteRepository repo = new RemoteRepository("remote", URI.create("https://localhost/"));
        // A relative MRL path, which is what every repository sets as the repository uri.
        Path resourceDir =
                repo.getResourceDirectory(artifact(URI.create("model/ai/djl/resnet/"), "0.0.1"));
        Assert.assertTrue(
                Utils.isInsideDirectory(repo.getCacheDirectory(), resourceDir),
                resourceDir + " is not inside the cache");
    }

    @Test
    public void testNestedItemNameResolves() throws IOException {
        // An item name carrying a directory is legitimate; only one that leaves the base is not.
        Path dir = ROOT.resolve("nested").toAbsolutePath();
        Files.createDirectories(dir.resolve("vocab"));
        Files.write(dir.resolve("vocab/tokens.txt"), "a".getBytes(StandardCharsets.UTF_8));
        SimpleRepository repo = new SimpleRepository("nested", dir.toUri(), dir);

        Artifact.Item item = new Artifact.Item();
        item.setName("vocab/tokens.txt");
        item.setArtifact(artifact(dir.toUri(), "0.0.1"));
        Assert.assertEquals(Paths.get(repo.resolvePath(item, "")), dir.resolve("vocab/tokens.txt"));
    }

    @Test
    public void testSingleFileArtifactResolvesToTheFile() throws IOException {
        // A single-file model resolves onto a regular file, which BaseModelLoader handles by
        // loading from its parent. Only the repository's own metadata.json is refused.
        RemoteRepository repo = new RemoteRepository("single", URI.create("https://localhost/"));
        Artifact artifact = artifact(URI.create("model/ai/djl/single/"), "m.onnx");
        Path resourceDir = repo.getResourceDirectory(artifact);
        Files.createDirectories(Objects.requireNonNull(resourceDir.getParent()));
        Files.write(resourceDir, "x".getBytes(StandardCharsets.UTF_8));
        try {
            // Resolving again is what every later prepare() and the load itself do.
            Assert.assertEquals(repo.getResourceDirectory(artifact), resourceDir);
        } finally {
            Utils.deleteQuietly(resourceDir);
        }
    }

    private static Artifact artifact(URI repositoryUri, String version) {
        Metadata metadata = new Metadata();
        metadata.setRepositoryUri(repositoryUri);
        Artifact artifact = new Artifact();
        artifact.setMetadata(metadata);
        artifact.setVersion(version);
        return artifact;
    }
}
