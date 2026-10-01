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

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator.SynsetLoader;
import ai.djl.ndarray.NDList;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.Utils;

import com.sun.net.httpserver.HttpServer;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/** Tests for the synset sources of {@link BaseImageTranslator}. */
public class SynsetLoaderTest {

    private String savedInsecureUrl;
    private String savedAllowOutside;
    private String savedOffline;
    private Path root;
    private Path modelDir;

    @BeforeMethod
    public void setUp() throws IOException {
        // The build forwards every ai.djl.* system property into the test JVM, so record these and
        // put them back in cleanup() rather than leaving them cleared for later tests.
        savedInsecureUrl = System.getProperty("ai.djl.allow_insecure_url");
        savedAllowOutside = System.getProperty("ai.djl.allow_files_outside_model_dir");
        savedOffline = System.getProperty("ai.djl.offline");
        for (String env :
                new String[] {
                    "DJL_ALLOW_INSECURE_URL", "DJL_ALLOW_FILES_OUTSIDE_MODEL_DIR", "DJL_OFFLINE"
                }) {
            if (Utils.getenv(env) != null) {
                throw new SkipException(env + " is set in the environment");
            }
        }
        System.clearProperty("ai.djl.allow_insecure_url");
        System.clearProperty("ai.djl.allow_files_outside_model_dir");
        // Offline mode is refused before the destination is examined, which would make the
        // loopback assertion below read "Offline mode is enabled" instead.
        System.clearProperty("ai.djl.offline");

        root = Files.createTempDirectory("djl-synset");
        modelDir = Files.createDirectories(root.resolve("model"));
        Files.write(modelDir.resolve("synset.txt"), "cat\ndog".getBytes(StandardCharsets.UTF_8));
        Files.write(root.resolve("outside.txt"), "outside".getBytes(StandardCharsets.UTF_8));
    }

    @AfterMethod
    public void cleanup() {
        restore("ai.djl.allow_insecure_url", savedInsecureUrl);
        restore("ai.djl.allow_files_outside_model_dir", savedAllowOutside);
        restore("ai.djl.offline", savedOffline);
        if (root != null) {
            Utils.deleteQuietly(root);
        }
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    // ----- synsetUrl -----

    @Test
    public void testLocalSynsetUrlIsRejected() throws IOException {
        URL file = root.resolve("outside.txt").toUri().toURL();
        URL jar = new URL("jar:" + file + "!/synset.txt");
        for (URL url : new URL[] {file, jar}) {
            IllegalArgumentException e =
                    Assert.expectThrows(
                            IllegalArgumentException.class,
                            () ->
                                    ImageClassificationTranslator.builder()
                                            .optSynsetUrl(url.toString()));
            Assert.assertTrue(
                    e.getMessage().contains("Unsupported synsetUrl protocol"),
                    url + " -> " + e.getMessage());
        }

        // The same check applies when synsetUrl comes from the model's arguments.
        Map<String, String> arguments = new HashMap<>();
        arguments.put("synsetUrl", file.toString());
        ImageClassificationTranslatorFactory factory = new ImageClassificationTranslatorFactory();
        try (Model model = Model.newInstance("test")) {
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            factory.newInstance(
                                    Image.class, Classifications.class, model, arguments));
        }
    }

    @Test
    public void testRemoteSynsetUrlIsAccepted() {
        // Nothing is fetched until the translator is prepared, so these only have to build.
        for (String url :
                new String[] {
                    "https://resources.djl.ai/synset.txt", "http://resources.djl.ai/synset.txt"
                }) {
            ImageClassificationTranslator translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new ToTensor())
                            .optSynsetUrl(url)
                            .build();
            Assert.assertNotNull(translator);
        }
    }

    @Test
    public void testSynsetUrlUsesUrlAccessChecks() throws IOException {
        // The fetch goes through Utils.openUrl, so a non-public destination is refused before any
        // request is made.
        AtomicInteger hits = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    hits.incrementAndGet();
                    byte[] out = "cat\ndog".getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(200, out.length);
                    exchange.getResponseBody().write(out);
                    exchange.close();
                });
        server.start();
        try {
            URL url = new URL("http://127.0.0.1:" + server.getAddress().getPort() + "/synset.txt");
            SynsetLoader loader = new SynsetLoader(url);
            IOException e = Assert.expectThrows(IOException.class, () -> loader.load(null));
            Assert.assertTrue(
                    e.getMessage().contains("non-public"), "unexpected: " + e.getMessage());
            Assert.assertEquals(hits.get(), 0, "the destination must not be contacted");
        } finally {
            server.stop(0);
        }
    }

    @Test
    public void testInsecureUrlOptOutAllowsLocalSynsetUrl() throws IOException {
        System.setProperty("ai.djl.allow_insecure_url", "true");
        URL url = root.resolve("outside.txt").toUri().toURL();
        List<String> synset = new SynsetLoader(url).load(null);
        Assert.assertEquals(synset, Collections.singletonList("outside"));
    }

    // ----- synsetFileName -----

    @Test
    public void testSynsetFileNameResolvesInModelDir() throws IOException, ModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(modelDir)
                        .optBlock(new LambdaBlock(a -> a, "model"))
                        .optEngine("PyTorch")
                        .optOption("hasParameter", "false")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            List<String> synset = new SynsetLoader("synset.txt").load(model);
            Assert.assertEquals(synset, Arrays.asList("cat", "dog"));

            SynsetLoader outside = new SynsetLoader("../outside.txt");
            IllegalArgumentException e =
                    Assert.expectThrows(IllegalArgumentException.class, () -> outside.load(model));
            Assert.assertTrue(
                    e.getMessage().contains("outside the model directory"),
                    "unexpected: " + e.getMessage());
        }
    }
}
