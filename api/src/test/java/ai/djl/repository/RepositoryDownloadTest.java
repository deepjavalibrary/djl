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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;

/** Tests the URL handling applied when downloading artifacts declared by repository metadata. */
public class RepositoryDownloadTest {

    @Test
    public void testNonPublicArtifactUriIsRejected() throws IOException {
        // An artifact URI declared in repository metadata may be absolute, which bypasses base-URI
        // resolution. It must still be subject to the same destination rule as the other download
        // paths, so a non-public destination is refused.
        IOException e = downloadItemUri("http://169.254.169.254/latest/meta-data/x.tar.gz");
        Assert.assertTrue(
                e.getMessage().contains("non-public"), "unexpected message: " + e.getMessage());
    }

    @Test
    public void testPrivateArtifactUriIsRejected() throws IOException {
        IOException e = downloadItemUri("http://10.0.0.5/model.tar.gz");
        Assert.assertTrue(
                e.getMessage().contains("non-public"), "unexpected message: " + e.getMessage());
    }

    @Test
    public void testLoopbackArtifactUriIsRejected() throws IOException {
        IOException e = downloadItemUri("http://127.0.0.1:1/model.tar.gz");
        Assert.assertTrue(
                e.getMessage().contains("non-public"), "unexpected message: " + e.getMessage());
    }

    @Test
    public void testUnsupportedSchemeArtifactUriIsRejected() throws IOException {
        IOException e = downloadItemUri("ftp://ftp.example.com/model.tar.gz");
        Assert.assertTrue(
                e.getMessage().contains("URL protocol"), "unexpected message: " + e.getMessage());
    }

    /**
     * Invokes the shared artifact-download path with an absolute item URI and returns the thrown
     * exception.
     */
    private IOException downloadItemUri(String itemUri) throws IOException {
        Path tmp = Files.createTempDirectory("djl-download-probe");
        try {
            Repository repository =
                    new RemoteRepository(
                            "test", URI.create("https://resources.djl.ai/test-models/"));
            Artifact.Item item = new Artifact.Item();
            item.setName("data");
            item.setUri(itemUri);

            return Assert.expectThrows(
                    IOException.class,
                    () ->
                            ((AbstractRepository) repository)
                                    .download(tmp, URI.create(""), item, null));
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void testSimpleUrlRepositoryRejectsNonPublicHost() {
        // The content-length HEAD probe is a second entry point into the same download flow, so it
        // applies the same destination rule.
        SimpleUrlRepository repository =
                new SimpleUrlRepository(
                        "test",
                        URI.create("http://169.254.169.254/latest/meta-data/x.tar.gz"),
                        "x.tar.gz");
        // getResources() resolves metadata, which performs the HEAD probe.
        Assert.assertTrue(repository.getResources().isEmpty());
    }

    @Test
    public void testSimpleUrlRepositoryDownloadRejectsNonPublicHost() throws IOException {
        SimpleUrlRepository repository =
                new SimpleUrlRepository(
                        "test", URI.create("http://10.0.0.5/model.tar.gz"), "model.tar.gz");
        Path tmp = Files.createTempDirectory("djl-simpleurl-probe");
        try {
            Artifact.Item item = new Artifact.Item();
            item.setName("data");
            item.setUri("http://10.0.0.5/model.tar.gz");
            IOException e =
                    Assert.expectThrows(
                            IOException.class,
                            () -> repository.download(tmp, URI.create(""), item, null));
            Assert.assertTrue(
                    e.getMessage().contains("non-public"), "unexpected: " + e.getMessage());
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
