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
package ai.djl.modality.cv.translator;

import ai.djl.modality.Input;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;

import com.sun.net.httpserver.HttpServer;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.concurrent.atomic.AtomicInteger;

/** Tests that an image URL arriving with a request is limited to data: and http(s) URLs. */
public class RequestImageUrlTest {

    // Written with forward slashes so that it is also a valid relative URI on Windows.
    private static final String IMAGE = "../examples/src/test/resources/kitten.jpg";

    private String savedInsecureUrl;

    @BeforeClass
    public void setUp() {
        if (Utils.getenv("DJL_ALLOW_INSECURE_URL") != null) {
            throw new SkipException("DJL_ALLOW_INSECURE_URL is set in the environment");
        }
        savedInsecureUrl = System.getProperty("ai.djl.allow_insecure_url");
        System.clearProperty("ai.djl.allow_insecure_url");
    }

    @AfterClass
    public void tearDown() {
        if (savedInsecureUrl == null) {
            System.clearProperty("ai.djl.allow_insecure_url");
        } else {
            System.setProperty("ai.djl.allow_insecure_url", savedInsecureUrl);
        }
    }

    @Test
    public void testDataUri() throws IOException {
        Path file = Paths.get(IMAGE);
        String url =
                "data:image/jpeg;base64,"
                        + Base64.getEncoder().encodeToString(Files.readAllBytes(file));
        Image expected = ImageFactory.getInstance().fromFile(file);
        Image image = ImageFactory.getInstance().fromRequestUrl(url);
        Assert.assertEquals(image.getWidth(), expected.getWidth());
        Assert.assertEquals(image.getHeight(), expected.getHeight());
    }

    @Test
    public void testLocalImageRejected() throws IOException {
        ImageFactory factory = ImageFactory.getInstance();
        String fileUrl = Paths.get(IMAGE).toAbsolutePath().toUri().toString();
        String[] urls = {
            IMAGE, fileUrl, "jar:" + fileUrl + "!/kitten.jpg", "ftp://localhost/a.jpg"
        };
        for (String url : urls) {
            Assert.assertThrows(IOException.class, () -> factory.fromRequestUrl(url));
        }

        // Application code can still load a local image with fromUrl.
        Assert.assertTrue(factory.fromUrl(fileUrl).getWidth() > 0);
    }

    @Test
    public void testPrivateHostRefusedBeforeConnecting() throws IOException {
        AtomicInteger hits = new AtomicInteger();
        HttpServer server = startServer(hits);
        try {
            String url = "http://127.0.0.1:" + server.getAddress().getPort() + "/kitten.jpg";
            Assert.assertThrows(
                    IOException.class, () -> ImageFactory.getInstance().fromRequestUrl(url));
            Assert.assertEquals(hits.get(), 0, "the destination must not be contacted");
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testAllowInsecureUrl() throws IOException {
        ImageFactory factory = ImageFactory.getInstance();
        String fileUrl = Paths.get(IMAGE).toAbsolutePath().toUri().toString();
        System.setProperty("ai.djl.allow_insecure_url", "true");
        try {
            Assert.assertTrue(factory.fromRequestUrl(fileUrl).getWidth() > 0);
            Assert.assertTrue(factory.fromRequestUrl(IMAGE).getWidth() > 0);
        } finally {
            System.clearProperty("ai.djl.allow_insecure_url");
        }
    }

    @Test
    public void testAllowInsecureUrlKeepsOfflineMode() throws IOException {
        if (Utils.getenv("DJL_OFFLINE") != null) {
            throw new SkipException("DJL_OFFLINE is set in the environment");
        }
        AtomicInteger hits = new AtomicInteger();
        HttpServer server = startServer(hits);
        String savedOffline = System.getProperty("ai.djl.offline");
        System.setProperty("ai.djl.allow_insecure_url", "true");
        System.setProperty("ai.djl.offline", "true");
        try {
            String url = "http://127.0.0.1:" + server.getAddress().getPort() + "/kitten.jpg";
            Assert.assertThrows(
                    IOException.class, () -> ImageFactory.getInstance().fromRequestUrl(url));
            Assert.assertEquals(hits.get(), 0, "nothing is fetched in offline mode");
        } finally {
            System.clearProperty("ai.djl.allow_insecure_url");
            if (savedOffline == null) {
                System.clearProperty("ai.djl.offline");
            } else {
                System.setProperty("ai.djl.offline", savedOffline);
            }
            server.stop(0);
        }
    }

    @Test
    public void testRequestInputs() {
        String fileUrl = Paths.get(IMAGE).toAbsolutePath().toUri().toString();

        Input input = new Input();
        input.addProperty("Content-Type", "application/json");
        input.add("{\"image_url\": \"" + fileUrl + "\"}");
        ImageServingTranslator translator = new ImageServingTranslator(new NoopTranslator());
        Assert.assertThrows(TranslateException.class, () -> translator.processInput(null, input));

        Input input2 = new Input();
        input2.add("{\"image\": \"" + fileUrl + "\", \"candidate_labels\": [\"cat\"]}");
        Assert.assertThrows(IOException.class, () -> VisionLanguageInput.parseInput(input2));

        String json =
                "{\"image_url\": \""
                        + fileUrl
                        + "\", \"prompt\": [{\"type\": \"point\", \"data\": [1, 1], \"label\":"
                        + " 0}]}";
        Assert.assertThrows(IOException.class, () -> Sam2Input.fromJson(json));
    }

    private static HttpServer startServer(AtomicInteger hits) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    hits.incrementAndGet();
                    exchange.sendResponseHeaders(404, -1);
                    exchange.close();
                });
        server.start();
        return server;
    }

    private static final class NoopTranslator implements Translator<Image, String> {

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            return new NDList();
        }

        /** {@inheritDoc} */
        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            return "";
        }
    }
}
