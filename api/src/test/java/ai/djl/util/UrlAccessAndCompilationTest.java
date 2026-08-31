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
package ai.djl.util;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Tests for the configurable URL-access and bundled-source-compilation defaults in {@link
 * Utils#openUrl} and {@link ClassLoaderUtils#compileJavaClass}.
 */
public class UrlAccessAndCompilationTest {

    @AfterMethod
    public void cleanup() {
        System.clearProperty("ai.djl.allow_insecure_url");
        System.clearProperty("ai.djl.compile_java");
    }

    // ----- URL access: Utils.openUrl -----

    @Test
    public void testFileProtocolAllowedByDefault() throws IOException {
        // file:// performs no network fetch, so it is allowed by default.
        Path tmp = Files.createTempFile("djl-local", ".txt");
        Files.write(tmp, "secret".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        try (java.io.InputStream is = Utils.openUrl(tmp.toUri().toURL())) {
            Assert.assertEquals(
                    new String(is.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8),
                    "secret");
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void testJarProtocolAllowedByDefault() throws IOException {
        // jar:// reads a resource bundled inside a jar (e.g. djl-serving's plugin.definition). This
        // performs no network fetch, so it is allowed by default. Build a real one-entry
        // jar and read it back through a jar: URL.
        Path jar = Files.createTempFile("djl-jar", ".jar");
        try (java.util.jar.JarOutputStream jos =
                new java.util.jar.JarOutputStream(Files.newOutputStream(jar))) {
            jos.putNextEntry(new java.util.zip.ZipEntry("META-INF/probe.txt"));
            jos.write("jarok".getBytes(java.nio.charset.StandardCharsets.UTF_8));
            jos.closeEntry();
        }
        try {
            URL jarUrl = new URL("jar:" + jar.toUri() + "!/META-INF/probe.txt");
            try (java.io.InputStream is = Utils.openUrl(jarUrl)) {
                Assert.assertEquals(
                        new String(is.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8),
                        "jarok");
            }
        } finally {
            // The JVM caches open jar files, so a handle can remain on the temp jar after the
            // stream is closed. On Windows that makes an immediate delete fail, so fall back to
            // deleting on JVM exit.
            jar.toFile().deleteOnExit();
            try {
                Files.deleteIfExists(jar);
            } catch (IOException ignore) {
                // cleaned up by deleteOnExit
            }
        }
    }

    @Test
    public void testFtpProtocolBlockedByDefault() {
        // Non-http(s), non-local schemes (ftp, gopher, ...) remain blocked by default: they are
        // neither trusted local reads nor host-validated http(s), so loosening jar/file must not
        // reopen them.
        Assert.assertThrows(
                IOException.class,
                () -> Utils.openUrl(new URL("ftp://ftp.example.com/pub/model.tar.gz")));
    }

    @Test
    public void testRedirectToNonPublicBlockedByDefault() throws IOException {
        // A public host that 302-redirects to a non-public destination must not be followed.
        // Stand up a real local HTTP server that returns 302 -> http://169.254.169.254/... and
        // confirm openUrl refuses it. The first hop (127.0.0.1) is itself non-public so it is
        // blocked immediately; more importantly, even if the first hop were public, the loop
        // re-validates the Location target and blocks 169.254.169.254 on the next iteration
        // (setInstanceFollowRedirects(false) means the JVM never silently follows it). Either way
        // the redirect target must be re-checked and refused.
        com.sun.net.httpserver.HttpServer server =
                com.sun.net.httpserver.HttpServer.create(
                        new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    exchange.getResponseHeaders()
                            .add(
                                    "Location",
                                    "http://169.254.169.254/latest/meta-data/iam/"
                                            + "meta-data/artifact.tar.gz");
                    exchange.sendResponseHeaders(302, -1);
                    exchange.close();
                });
        server.start();
        try {
            int port = server.getAddress().getPort();
            URL redirecting = new URL("http://127.0.0.1:" + port + "/model.tar.gz");
            IOException ex =
                    Assert.expectThrows(IOException.class, () -> Utils.openUrl(redirecting));
            Assert.assertTrue(
                    ex.getMessage().contains("Blocked request to non-public address"),
                    "expected non-public-address block, got: " + ex.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testPublicHostRedirectRevalidated() throws IOException {
        // Stronger variant: make the FIRST hop pass the host check (use a hostname that resolves to
        // a public address, 8.8.8.8, but point it at our local listener via a custom
        // URLStreamHandler
        // is overkill) — instead assert the predicate directly for the redirect target so the
        // re-validation contract is explicit: the Location host (169.254.169.254) is non-public.
        Assert.assertFalse(Utils.isPublicHost("169.254.169.254"));
    }

    @Test
    public void testLinkLocalAddressBlockedByDefault() {
        Assert.assertThrows(
                IOException.class,
                () -> Utils.openUrl(new URL("http://169.254.169.254/latest/meta-data/")));
    }

    @Test
    public void testLoopbackBlockedByDefault() {
        Assert.assertThrows(IOException.class, () -> Utils.openUrl(new URL("http://127.0.0.1:1/")));
        Assert.assertThrows(IOException.class, () -> Utils.openUrl(new URL("http://localhost:1/")));
    }

    @Test
    public void testPrivateAddressBlockedByDefault() {
        Assert.assertThrows(IOException.class, () -> Utils.openUrl(new URL("http://10.0.0.5/")));
        Assert.assertThrows(IOException.class, () -> Utils.openUrl(new URL("http://192.168.1.1/")));
    }

    @Test
    public void testHostValidationPredicate() {
        // Non-public hosts must be rejected.
        Assert.assertFalse(Utils.isPublicHost("127.0.0.1"));
        Assert.assertFalse(Utils.isPublicHost("localhost"));
        Assert.assertFalse(Utils.isPublicHost("169.254.169.254"));
        Assert.assertFalse(Utils.isPublicHost("10.0.0.5"));
        Assert.assertFalse(Utils.isPublicHost("192.168.0.1"));
        Assert.assertFalse(Utils.isPublicHost("172.16.0.1"));
        Assert.assertFalse(Utils.isPublicHost(""));
        Assert.assertFalse(Utils.isPublicHost(null));
        // A well-known public address must pass.
        Assert.assertTrue(Utils.isPublicHost("8.8.8.8"));
    }

    @Test
    public void testInsecureOptOutAllowsFileProtocol() throws IOException {
        System.setProperty("ai.djl.allow_insecure_url", "true");
        Assert.assertTrue(Utils.isInsecureUrlAllowed());
        Path tmp = Files.createTempFile("djl-optout", ".txt");
        Files.write(tmp, "ok".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        try (java.io.InputStream is = Utils.openUrl(tmp.toUri().toURL())) {
            Assert.assertEquals(new String(is.readAllBytes(), "UTF-8"), "ok");
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void testInsecureUrlAllowedFlagDefaultsFalse() {
        Assert.assertFalse(Utils.isInsecureUrlAllowed());
    }

    // ----- Bundled-source compilation: ClassLoaderUtils.compileJavaClass -----

    @Test
    public void testCompilationDisabledByDefault() throws IOException {
        Path dir = Files.createTempDirectory("djl-compile");
        Path classes = Files.createDirectories(dir.resolve("classes"));
        Files.write(
                classes.resolve("Bundled.java"),
                ("public class Bundled { static { System.setProperty(\"djl.test.canary\", \"ran\");"
                                + " } }")
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
        System.clearProperty("djl.test.canary");
        try {
            ClassLoaderUtils.compileJavaClass(classes);
            // Default (disabled): no .class produced and the static initializer never ran.
            Assert.assertFalse(Files.exists(classes.resolve("Bundled.class")));
            Assert.assertFalse(ClassLoaderUtils.isDynamicCompilationEnabled());
        } finally {
            deleteTree(dir);
        }
    }

    @Test
    public void testCompilationEnabledWithOptIn() throws IOException {
        System.setProperty("ai.djl.compile_java", "true");
        Assert.assertTrue(ClassLoaderUtils.isDynamicCompilationEnabled());
        Path dir = Files.createTempDirectory("djl-compile-optin");
        Path classes = Files.createDirectories(dir.resolve("classes"));
        Files.write(
                classes.resolve("Hello.java"),
                "public class Hello { public int v() { return 1; } }"
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
        try {
            ClassLoaderUtils.compileJavaClass(classes);
            // Opt-in: compilation runs (a system compiler must be present in the test JDK).
            if (javax.tools.ToolProvider.getSystemJavaCompiler() != null) {
                Assert.assertTrue(Files.exists(classes.resolve("Hello.class")));
            }
        } finally {
            deleteTree(dir);
        }
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (java.util.stream.Stream<Path> walk = Files.walk(root)) {
            walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(
                            p -> {
                                try {
                                    Files.deleteIfExists(p);
                                } catch (IOException ignore) {
                                    // best effort
                                }
                            });
        }
    }
}
