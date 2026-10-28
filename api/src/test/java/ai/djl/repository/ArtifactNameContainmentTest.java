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

import ai.djl.Application;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class ArtifactNameContainmentTest {

    private static final Path DIR = Paths.get("build/containment");

    @BeforeClass
    public void setUp() {
        // Absolute: these resolve cache paths, and ZipUtils refuses every entry under a relative
        // destination. Without it the cache is the real ~/.djl.ai.
        System.setProperty("DJL_CACHE_DIR", Paths.get("build/cache").toAbsolutePath().toString());
    }

    @AfterClass
    public void tearDown() {
        System.clearProperty("DJL_CACHE_DIR");
    }

    @Test
    public void testNameStayingInsideIsResolved() throws IOException {
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, "model.onnx"),
                DIR.resolve("model.onnx").toAbsolutePath().normalize());
        // A nested name is legal: the dir branch of save() exists to honour it.
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, "sub/dir/model.onnx"),
                DIR.resolve("sub/dir/model.onnx").toAbsolutePath().normalize());
        // An empty name is deliberate in SimpleRepository and RpcRepository, and resolves to the
        // directory itself.
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, ""), DIR.toAbsolutePath().normalize());
        // Interior traversal that does not escape is fine.
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, "sub/../model.onnx"),
                DIR.resolve("model.onnx").toAbsolutePath().normalize());
    }

    @Test
    public void testNameEscapingIsRefused() {
        Assert.assertThrows(
                IOException.class, () -> AbstractRepository.resolveInside(DIR, "../escaped.onnx"));
        Assert.assertThrows(
                IOException.class,
                () -> AbstractRepository.resolveInside(DIR, "a/../../../etc/passwd"));
        // An absolute name ignores the directory entirely.
        Assert.assertThrows(
                IOException.class, () -> AbstractRepository.resolveInside(DIR, "/etc/passwd"));
        // A missing name is refused rather than resolving to the directory.
        Assert.assertThrows(IOException.class, () -> AbstractRepository.resolveInside(DIR, null));
    }

    @Test
    public void testEscapingVersionIsRefusedNotRerooted() throws IOException {
        // URI.relativize hands back its argument when the result is not a descendant, so without
        // a check the escape is re-rooted under the repository directory instead of refused.
        Repository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        Artifact artifact = artifactOf("model/undefined/com/example/tiny/", "../x", "");
        IOException e =
                Assert.expectThrows(IOException.class, () -> repo.getResourceDirectory(artifact));
        Assert.assertTrue(
                e.getMessage().contains("outside the repository"), "unexpected: " + e.getMessage());
    }

    @Test
    public void testVersionNamingAnExistingFileIsRefused() throws IOException {
        // The repository's own cached metadata is a file in this tree. Resolving onto it makes
        // prepare() treat the artifact as downloaded and the model directory become its parent.
        Repository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        Path mrlDir = repo.getCacheDirectory().resolve("model/undefined/com/example/tiny");
        try {
            Files.createDirectories(mrlDir);
            Files.write(mrlDir.resolve("metadata.json"), "{}".getBytes(StandardCharsets.UTF_8));

            Artifact artifact =
                    artifactOf("model/undefined/com/example/tiny/", "metadata.json", "");
            IOException e =
                    Assert.expectThrows(
                            IOException.class, () -> repo.getResourceDirectory(artifact));
            Assert.assertTrue(
                    e.getMessage().contains("repository metadata"),
                    "unexpected: " + e.getMessage());
        } finally {
            // Written into the real cache: a "{}" metadata.json left here makes a later
            // RemoteRepository.locate of this MRL fail on a null lastUpdated.
            Utils.deleteQuietly(mrlDir);
        }
    }

    @Test
    public void testSynthesizedRepositoryWithNoVersionOrNameResolvesToItsOwnDir()
            throws IOException {
        // A synthesized repository derives the name from the URL, so an hdfs:// root or an empty
        // model_name gives an empty name and the repository's own directory. It caches no
        // metadata there, so there is nothing for an item to replace.
        Repository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        Artifact artifact = artifactOf("model/cv/ai/victim/", null, "");
        artifact.getMetadata().setRepositoryUri(URI.create("model/cv/ai/victim/"));
        Assert.assertEquals(
                repo.getResourceDirectory(matchAll(artifact)),
                repo.getCacheDirectory().resolve("model/cv/ai/victim"));
    }

    @Test
    public void testMetadataBackedArtifactCannotResolveToTheRepositoryDir() {
        // The repository's own metadata.json lives in that directory, so an artifact resolving
        // onto it can have an item replace the metadata every later load of this MRL reads.
        Repository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        Artifact artifact = artifactOf("model/cv/ai/victim/", null, "");
        IOException e =
                Assert.expectThrows(IOException.class, () -> repo.getResourceDirectory(artifact));
        Assert.assertTrue(
                e.getMessage().contains("not inside it"), "wrong refusal: " + e.getMessage());
    }

    private static Artifact matchAll(Artifact artifact) {
        Metadata.MatchAllMetadata metadata = new Metadata.MatchAllMetadata();
        metadata.setRepositoryUri(artifact.getMetadata().getRepositoryUri());
        artifact.setMetadata(metadata);
        return artifact;
    }

    @Test
    public void testTraversingNameCannotCollapseOntoAnotherMrl() throws IOException {
        // Containing only to the cache root would let a traversing name land on a different MRL's
        // directory. A repository that simply declares that MRL still reaches it: the cache is
        // keyed by MRL alone, which this does not change.
        Repository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        Path victim =
                repo.getResourceDirectory(
                        artifactOf(
                                "model/cv/image_classification/ai/djl/pytorch/resnet/",
                                "0.0.1",
                                "traced_resnet18"));
        Artifact attacker =
                artifactOf(
                        "model/undefined/com/example/tiny/",
                        "0.0.1",
                        "../../../../../cv/image_classification/ai/djl/pytorch/resnet/0.0.1"
                                + "/traced_resnet18");
        Path reached = null;
        try {
            reached = repo.getResourceDirectory(attacker);
        } catch (IOException expected) {
            return;
        }
        Assert.assertNotEquals(reached, victim, "reached another model's cache directory");
    }

    private static Artifact artifactOf(String repositoryUri, String version, String name) {
        Metadata metadata = new Metadata();
        metadata.setRepositoryUri(URI.create(repositoryUri));
        Artifact artifact = new Artifact();
        artifact.setMetadata(metadata);
        artifact.setVersion(version);
        artifact.setName(name);
        return artifact;
    }

    @Test
    public void testMrlEscapingCacheIsRefused() throws IOException {
        // locate creates the metadata cache directory from the MRL path before downloading.
        Repository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        MRL mrl = repo.model(Application.UNDEFINED, "ai.djl", "../../../../../escaped");
        // The message and the absent directory, not just any IOException: the fetch to localhost
        // throws one of its own, so assertThrows alone passes with the containment removed.
        // Cleared first: the escape lands beside the repo subdirectory, and a run without the
        // containment leaves it behind.
        Path escaped =
                Objects.requireNonNull(repo.getCacheDirectory().getParent()).resolve("escaped");
        Utils.deleteQuietly(escaped);
        IOException e = Assert.expectThrows(IOException.class, () -> repo.locate(mrl));
        Assert.assertTrue(
                e.getMessage().contains("resolves outside"), "unexpected: " + e.getMessage());
        Assert.assertFalse(Files.exists(escaped), escaped + " was created");
    }

    @Test
    public void testItemNameEscapingSimpleRepositoryIsRefused() throws IOException {
        // A local SimpleRepository resolves the item name itself rather than through the base
        // class, so the containment has to be applied there too.
        Path dir = Paths.get("build/containment");
        Files.createDirectories(dir);
        SimpleRepository repo = new SimpleRepository("test", dir.toUri(), dir);
        Artifact artifact = new Artifact();
        artifact.setName("escaping");
        Artifact.Item item = new Artifact.Item();
        item.setName("../../../../../escaped.onnx");
        item.setArtifact(artifact);
        Assert.assertThrows(IOException.class, () -> repo.resolvePath(item, ""));
    }

    @Test
    public void testAbsoluteFileItemUriIsRefusedForARemoteRepository() throws IOException {
        // download() reads the item uri and saves it as a model file, and Utils.openUrl treats
        // file: as a local read, so an absolute one in remote metadata reads an arbitrary file.
        Path buildDir = Paths.get("build");
        Files.createDirectories(buildDir);
        Path secret = buildDir.resolve("containment-secret.txt");
        Files.write(secret, "TOPSECRET".getBytes(StandardCharsets.UTF_8));
        Path dest = buildDir.resolve("containment-dl");
        Utils.deleteQuietly(dest);
        Files.createDirectories(dest);

        RemoteRepository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        Artifact.Item item = new Artifact.Item();
        item.setName("synset.txt");
        item.setUri(secret.toAbsolutePath().toUri().toString());

        IOException e =
                Assert.expectThrows(
                        IOException.class,
                        () -> repo.download(dest, URI.create("model/"), item, null));
        // The message pins the reason: without it the assertion passes on any download failure.
        Assert.assertTrue(
                e.getMessage().contains("names a local file"), "unexpected: " + e.getMessage());
        Assert.assertFalse(Files.exists(dest.resolve("synset.txt")));
    }

    @Test
    public void testItemUriSchemeIsMatchedWhateverItsCase() throws IOException {
        // URI.getScheme() keeps the case the metadata wrote, while URL.getProtocol() lower cases
        // it, so an upper case scheme passed a case-sensitive test here and still reached
        // Utils.openUrl as a local read.
        Path dest = downloadDir("containment-dl-case");
        Path secret = secretFile();

        RemoteRepository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        for (String scheme : new String[] {"FILE", "File"}) {
            Artifact.Item item = new Artifact.Item();
            item.setName("synset.txt");
            String uri = secret.toAbsolutePath().toUri().toString();
            item.setUri(scheme + uri.substring(uri.indexOf(':')));

            IOException e =
                    Assert.expectThrows(
                            IOException.class,
                            () -> repo.download(dest, URI.create("model/"), item, null));
            Assert.assertTrue(
                    e.getMessage().contains("names a local file"), "unexpected: " + e.getMessage());
            Assert.assertFalse(Files.exists(dest.resolve("synset.txt")));
        }
    }

    @Test
    public void testJarItemUriIsRefusedForARemoteRepository() throws IOException {
        // Utils.openUrl accepts a jar: URL whose nested URL is local, and the containment above
        // cannot judge it: the path that decides the read sits inside the nested URL, not in this
        // one. So the scheme is refused outright.
        Path dest = downloadDir("containment-dl-jar");
        Path archive = secretArchive();

        RemoteRepository repo = new RemoteRepository("test", URI.create("https://localhost/"));
        Artifact.Item item = new Artifact.Item();
        item.setName("synset.txt");
        item.setUri("jar:" + archive.toAbsolutePath().toUri() + "!/secret.txt");

        IOException e =
                Assert.expectThrows(
                        IOException.class,
                        () -> repo.download(dest, URI.create("model/"), item, null));
        Assert.assertTrue(
                e.getMessage().contains("unsupported scheme"), "unexpected: " + e.getMessage());
        Assert.assertFalse(Files.exists(dest.resolve("synset.txt")));
    }

    @Test
    public void testUppercaseFileItemUriEscapingALocalRepositoryIsRefused() throws IOException {
        // A local repository may read its own files, so this one is refused by the containment
        // rather than by the scheme, which an upper case scheme also used to skip.
        Path dir = Paths.get("build/containment");
        Files.createDirectories(dir);
        Path dest = downloadDir("containment-dl-local-case");
        Path secret = secretFile();

        LocalRepository repo = new LocalRepository("test", dir.toUri(), dir);
        Artifact.Item item = new Artifact.Item();
        item.setName("synset.txt");
        String uri = secret.toAbsolutePath().toUri().toString();
        item.setUri("FILE" + uri.substring(uri.indexOf(':')));

        IOException e =
                Assert.expectThrows(
                        IOException.class,
                        () -> repo.download(dest, URI.create("model/"), item, null));
        Assert.assertTrue(
                e.getMessage().contains("resolves outside"), "unexpected: " + e.getMessage());
        Assert.assertFalse(Files.exists(dest.resolve("synset.txt")));
    }

    private static Path downloadDir(String name) throws IOException {
        Path dest = Paths.get("build").resolve(name);
        Utils.deleteQuietly(dest);
        Files.createDirectories(dest);
        return dest;
    }

    private static Path secretFile() throws IOException {
        Path buildDir = Paths.get("build");
        Files.createDirectories(buildDir);
        Path secret = buildDir.resolve("containment-secret.txt");
        Files.write(secret, "TOPSECRET".getBytes(StandardCharsets.UTF_8));
        return secret;
    }

    /** A real archive, so the read is refused by the scheme rather than by failing to parse. */
    private static Path secretArchive() throws IOException {
        Path buildDir = Paths.get("build");
        Files.createDirectories(buildDir);
        Path archive = buildDir.resolve("containment-secret.jar");
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(archive))) {
            zos.putNextEntry(new ZipEntry("secret.txt"));
            zos.write("TOPSECRET".getBytes(StandardCharsets.UTF_8));
            zos.closeEntry();
        }
        return archive;
    }

    @Test
    public void testItemUriEscapingLocalRepositoryIsRefused() throws IOException {
        // A local repository takes the metadata-item branch, where the item's own uri is resolved
        // against the artifact rather than against a cache directory.
        Path dir = Paths.get("build/containment");
        Files.createDirectories(dir);
        LocalRepository repo = new LocalRepository("test", dir.toUri(), dir);
        Metadata metadata = new Metadata();
        metadata.setRepositoryUri(URI.create("model/ai/djl/escaping/"));
        Artifact artifact = new Artifact();
        artifact.setMetadata(metadata);
        artifact.setVersion("0.0.1");
        Artifact.Item item = new Artifact.Item();
        item.setName("synset.txt");
        item.setUri("../../../../../../etc/passwd");
        item.setArtifact(artifact);
        Assert.assertThrows(IOException.class, () -> repo.resolvePath(item, ""));
    }

    @Test
    public void testMrlEscapingLocalRepositoryIsRefused() {
        // The same MRL path reaches LocalRepository.locate, which reads metadata.json from it.
        Path dir = Paths.get("build/containment");
        Repository repo = new LocalRepository("test", dir.toUri(), dir);
        MRL mrl = repo.model(Application.UNDEFINED, "ai.djl", "../../../../../escaped");
        Assert.assertThrows(IOException.class, () -> repo.locate(mrl));
    }
}
