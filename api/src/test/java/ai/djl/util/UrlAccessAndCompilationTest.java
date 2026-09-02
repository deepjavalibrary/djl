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

import com.sun.net.httpserver.HttpServer;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;

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
        Files.write(tmp, "secret".getBytes(StandardCharsets.UTF_8));
        try (InputStream is = Utils.openUrl(tmp.toUri().toURL())) {
            Assert.assertEquals(new String(is.readAllBytes(), StandardCharsets.UTF_8), "secret");
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
        try (JarOutputStream jos = new JarOutputStream(Files.newOutputStream(jar))) {
            jos.putNextEntry(new ZipEntry("META-INF/probe.txt"));
            jos.write("jarok".getBytes(StandardCharsets.UTF_8));
            jos.closeEntry();
        }
        try {
            URL jarUrl = new URL("jar:" + jar.toUri() + "!/META-INF/probe.txt");
            try (InputStream is = Utils.openUrl(jarUrl)) {
                Assert.assertEquals(new String(is.readAllBytes(), StandardCharsets.UTF_8), "jarok");
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
    public void testNonPublicHostBlockedBeforeConnecting() throws IOException {
        // The destination check runs before any request, so the listener must never be contacted.
        // Redirect re-validation is covered by testRedirectToDisallowedHostIsRejected.
        AtomicInteger hits = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    hits.incrementAndGet();
                    exchange.sendResponseHeaders(200, -1);
                    exchange.close();
                });
        server.start();
        try {
            URL url =
                    new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/model.tar.gz");
            IOException ex = Assert.expectThrows(IOException.class, () -> Utils.openUrl(url));
            Assert.assertTrue(
                    ex.getMessage().contains("Blocked request to non-public address"),
                    "expected non-public-address block, got: " + ex.getMessage());
            Assert.assertEquals(hits.get(), 0, "the destination must not be contacted");
        } finally {
            server.stop(0);
        }
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
    public void testInsecureUrlAllowedFlagDefaultsFalse() {
        Assert.assertFalse(Utils.isInsecureUrlAllowed());
    }

    // ----- Bundled-source compilation: ClassLoaderUtils.compileJavaClass -----

    @Test
    public void testCompilationDisabledByDefault() throws IOException {
        // The environment variable takes precedence over the system property, matching
        // DJL_OFFLINE, so skip rather than fail if a build host has it set.
        if (Utils.getenv("DJL_COMPILE_JAVA") != null) {
            throw new org.testng.SkipException("DJL_COMPILE_JAVA is set in the environment");
        }
        Path dir = Files.createTempDirectory("djl-compile");
        Path classes = Files.createDirectories(dir.resolve("classes"));
        Files.write(
                classes.resolve("Bundled.java"),
                ("public class Bundled { static { System.setProperty(\"djl.test.canary\", \"ran\");"
                                + " } }")
                        .getBytes(StandardCharsets.UTF_8));
        System.clearProperty("djl.test.canary");
        try {
            Assert.assertThrows(
                    IllegalStateException.class, () -> ClassLoaderUtils.compileJavaClass(classes));
            // Default (disabled): nothing compiled and the static initializer never ran.
            Assert.assertFalse(Files.exists(classes.resolve("Bundled.class")));
            Assert.assertNull(
                    System.getProperty("djl.test.canary"),
                    "the bundled static initializer must not have run");
            Assert.assertFalse(ClassLoaderUtils.isDynamicCompilationEnabled());
        } finally {
            deleteTree(dir);
        }
    }

    @Test
    public void testCompilationEnabledWithOptIn() throws IOException {
        if (Utils.getenv("DJL_COMPILE_JAVA") != null) {
            throw new org.testng.SkipException("DJL_COMPILE_JAVA is set in the environment");
        }
        System.setProperty("ai.djl.compile_java", "true");
        Assert.assertTrue(ClassLoaderUtils.isDynamicCompilationEnabled());
        Path dir = Files.createTempDirectory("djl-compile-optin");
        Path classes = Files.createDirectories(dir.resolve("classes"));
        Files.write(
                classes.resolve("Hello.java"),
                "public class Hello { public int v() { return 1; } }"
                        .getBytes(StandardCharsets.UTF_8));
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

    // ----- Redirect handling (exercised against a local server via the host-check seam) -----

    /** Serves a redirect chain / final body on loopback for redirect tests. */
    private static HttpServer startServer(Map<String, String> redirects, String body)
            throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    String path = exchange.getRequestURI().getPath();
                    String target = redirects.get(path);
                    if (target != null) {
                        if (!target.isEmpty()) {
                            exchange.getResponseHeaders().add("Location", target);
                        }
                        exchange.sendResponseHeaders(302, -1);
                    } else {
                        byte[] out = body.getBytes(StandardCharsets.UTF_8);
                        exchange.sendResponseHeaders(200, out.length);
                        exchange.getResponseBody().write(out);
                    }
                    exchange.close();
                });
        server.start();
        return server;
    }

    @Test
    public void testRedirectIsFollowedAndRevalidated() throws IOException {
        Map<String, String> r = new HashMap<>();
        r.put("/start", "/final");
        HttpServer server = startServer(r, "arrived");
        try {
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/start");
            // allow loopback so the redirect loop itself runs
            try (InputStream is =
                    Utils.openHttpConnection(url, "GET", Collections.emptyMap(), h -> true)
                            .getInputStream()) {
                Assert.assertEquals(
                        new String(is.readAllBytes(), StandardCharsets.UTF_8), "arrived");
            }
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testRedirectToDisallowedHostIsRejected() throws IOException {
        Map<String, String> r = new HashMap<>();
        r.put("/start", "http://169.254.169.254/latest/meta-data/x.tar.gz");
        HttpServer server = startServer(r, "unused");
        try {
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/start");
            // first hop allowed, redirect target must still be re-checked and refused
            IOException e =
                    Assert.expectThrows(
                            IOException.class,
                            () ->
                                    Utils.openHttpConnection(
                                            url,
                                            "GET",
                                            Collections.emptyMap(),
                                            h -> !"169.254.169.254".equals(h)));
            Assert.assertTrue(
                    e.getMessage().contains("non-public"), "unexpected: " + e.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testRedirectToUnsupportedSchemeIsRejected() throws IOException {
        Map<String, String> r = new HashMap<>();
        r.put("/start", "ftp://ftp.example.com/model.tar.gz");
        HttpServer server = startServer(r, "unused");
        try {
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/start");
            IOException e =
                    Assert.expectThrows(
                            IOException.class,
                            () ->
                                    Utils.openHttpConnection(
                                            url, "GET", Collections.emptyMap(), h -> true));
            Assert.assertTrue(
                    e.getMessage().contains("URL protocol"), "unexpected: " + e.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testRedirectWithoutLocationIsRejected() throws IOException {
        Map<String, String> r = new HashMap<>();
        r.put("/start", "");
        HttpServer server = startServer(r, "unused");
        try {
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/start");
            IOException e =
                    Assert.expectThrows(
                            IOException.class,
                            () ->
                                    Utils.openHttpConnection(
                                            url, "GET", Collections.emptyMap(), h -> true));
            Assert.assertTrue(
                    e.getMessage().contains("no Location"), "unexpected: " + e.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testTooManyRedirectsIsRejected() throws IOException {
        Map<String, String> r = new HashMap<>();
        for (int i = 0; i < 10; ++i) {
            r.put("/hop" + i, "/hop" + (i + 1));
        }
        HttpServer server = startServer(r, "unused");
        try {
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/hop0");
            IOException e =
                    Assert.expectThrows(
                            IOException.class,
                            () ->
                                    Utils.openHttpConnection(
                                            url, "GET", Collections.emptyMap(), h -> true));
            Assert.assertTrue(
                    e.getMessage().contains("Too many redirects"), "unexpected: " + e.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testInsecureOptOutAllowsNonPublicHttpHost() throws IOException {
        HttpServer server = startServer(new HashMap<>(), "opted-out");
        try {
            System.setProperty("ai.djl.allow_insecure_url", "true");
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/data");
            try (InputStream is = Utils.openUrl(url)) {
                Assert.assertEquals(
                        new String(is.readAllBytes(), StandardCharsets.UTF_8), "opted-out");
            }
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testUnknownHostIsNotPublic() {
        Assert.assertFalse(Utils.isPublicHost("no-such-host.invalid"));
    }

    @Test
    public void testIpv6UniqueLocalAddressIsNotPublic() {
        // fc00::/7 is the IPv6 unique-local range. InetAddress#isSiteLocalAddress only covers the
        // deprecated fec0::/10, so these must be rejected explicitly.
        Assert.assertFalse(Utils.isPublicHost("fd00::1"));
        Assert.assertFalse(Utils.isPublicHost("fc00::1"));
        Assert.assertFalse(Utils.isPublicHost("fdff:ffff::1"));
        // link-local and loopback IPv6 remain rejected, and a public IPv6 address is allowed
        Assert.assertFalse(Utils.isPublicHost("fe80::1"));
        Assert.assertFalse(Utils.isPublicHost("::1"));
        Assert.assertTrue(Utils.isPublicHost("2001:4860:4860::8888"));
    }

    @Test
    public void testOptOutAppliesToHeadConnection() throws IOException {
        // The shared connection helper must honor the opt-out too, otherwise callers that only need
        // response metadata (the content-length probe) stay blocked when the flag is set.
        HttpServer server = startServer(new HashMap<>(), "head-ok");
        try {
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/data");
            // blocked by default: loopback is not public
            Assert.assertThrows(
                    IOException.class,
                    () -> Utils.openHttpConnection(url, "HEAD", Collections.emptyMap()));
            System.setProperty("ai.djl.allow_insecure_url", "true");
            HttpURLConnection conn = Utils.openHttpConnection(url, "HEAD", Collections.emptyMap());
            try {
                Assert.assertEquals(conn.getResponseCode(), 200);
            } finally {
                conn.disconnect();
            }
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testSameOriginTreatsImplicitAndExplicitDefaultPortAsEqual()
            throws MalformedURLException {
        // URL#getPort returns -1 for an implicit port, so comparing raw ports would classify a
        // redirect that only makes the default port explicit as cross-origin and drop credentials.
        Assert.assertTrue(
                Utils.sameOrigin(
                        new URL("https://example.com/a"), new URL("https://example.com:443/b")));
        Assert.assertTrue(
                Utils.sameOrigin(
                        new URL("http://example.com/a"), new URL("http://example.com:80/b")));
        // A genuinely different port, scheme or host is still cross-origin.
        Assert.assertFalse(
                Utils.sameOrigin(
                        new URL("https://example.com/a"), new URL("https://example.com:8443/b")));
        Assert.assertFalse(
                Utils.sameOrigin(
                        new URL("https://example.com/a"), new URL("http://example.com/b")));
        Assert.assertFalse(
                Utils.sameOrigin(
                        new URL("https://example.com/a"), new URL("https://other.example/b")));
    }

    @Test
    public void testCrossOriginRedirectDropsCredentialsAndAllowsNullHeaderValue()
            throws IOException {
        // Two things at once, because they exercise the same hop: the credential header must not
        // reach the redirect target, and a caller's header map may legally hold a null value.
        Map<String, String> r = new HashMap<>();
        Map<String, String> seen = new HashMap<>();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        int port = server.getAddress().getPort();
        // 127.0.0.1 -> localhost is a different host string, so the hop is cross-origin.
        r.put("/start", "http://localhost:" + port + "/final");
        server.createContext(
                "/",
                exchange -> {
                    String path = exchange.getRequestURI().getPath();
                    String target = r.get(path);
                    if (target != null) {
                        exchange.getResponseHeaders().add("Location", target);
                        exchange.sendResponseHeaders(302, -1);
                    } else {
                        // Record what the final hop actually received.
                        seen.put(
                                "auth",
                                String.valueOf(
                                        exchange.getRequestHeaders().getFirst("Authorization")));
                        seen.put(
                                "keep",
                                String.valueOf(exchange.getRequestHeaders().getFirst("X-Keep")));
                        byte[] out = "arrived".getBytes(StandardCharsets.UTF_8);
                        exchange.sendResponseHeaders(200, out.length);
                        exchange.getResponseBody().write(out);
                    }
                    exchange.close();
                });
        server.start();
        try {
            Map<String, String> headers = new HashMap<>();
            headers.put("Authorization", "Bearer secret");
            headers.put("X-Keep", "kept");
            headers.put("X-Optional", null);
            URL url = new URL("http://127.0.0.1:" + port + "/start");
            try (InputStream is =
                    Utils.openHttpConnection(url, "GET", headers, h -> true).getInputStream()) {
                Assert.assertEquals(
                        new String(is.readAllBytes(), StandardCharsets.UTF_8), "arrived");
            }
            Assert.assertEquals(seen.get("auth"), "null", "credentials must not cross origins");
            Assert.assertEquals(seen.get("keep"), "kept", "other headers must still be sent");
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testSameOriginRedirectKeepsCredentials() throws IOException {
        // The counterpart to the test above: a same-origin hop must not lose the header.
        Map<String, String> r = new HashMap<>();
        Map<String, String> seen = new HashMap<>();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        r.put("/start", "/final");
        server.createContext(
                "/",
                exchange -> {
                    String target = r.get(exchange.getRequestURI().getPath());
                    if (target != null) {
                        exchange.getResponseHeaders().add("Location", target);
                        exchange.sendResponseHeaders(302, -1);
                    } else {
                        seen.put(
                                "auth",
                                String.valueOf(
                                        exchange.getRequestHeaders().getFirst("Authorization")));
                        exchange.sendResponseHeaders(200, -1);
                    }
                    exchange.close();
                });
        server.start();
        try {
            Map<String, String> headers = new HashMap<>();
            headers.put("Authorization", "Bearer secret");
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/start");
            HttpURLConnection conn = Utils.openHttpConnection(url, "GET", headers, h -> true);
            try {
                Assert.assertEquals(conn.getResponseCode(), 200);
            } finally {
                conn.disconnect();
            }
            Assert.assertEquals(seen.get("auth"), "Bearer secret");
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testCompilationSkippedQuietlyWithoutBundledSources() throws IOException {
        // No bundled sources is the common case and must be a silent no-op.
        Path dir = Files.createTempDirectory("djl-nosrc");
        Path classes = Files.createDirectories(dir.resolve("classes"));
        try {
            ClassLoaderUtils.compileJavaClass(classes);
            Assert.assertEquals(classes.toFile().list().length, 0);
        } finally {
            Files.deleteIfExists(classes);
            Files.deleteIfExists(dir);
        }
    }
}
