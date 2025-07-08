/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.genai.gemini;

import ai.djl.ModelException;
import ai.djl.genai.gemini.types.GenerationConfig;
import ai.djl.genai.gemini.types.GoogleSearch;
import ai.djl.genai.gemini.types.HarmBlockThreshold;
import ai.djl.genai.gemini.types.HarmCategory;
import ai.djl.genai.gemini.types.SafetySetting;
import ai.djl.genai.gemini.types.Tool;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestServer;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class GeminiTest {

    private static final Logger logger = LoggerFactory.getLogger(GeminiTest.class);

    @Test
    public void testGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String endpoint = "http://localhost:" + server.getPort();
            // endpoint = "https://generativelanguage.googleapis.com";
            String url = endpoint + "/v1beta/models/gemini-2.5-flash:generateContent";
            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234")
                            .build();

            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in = GeminiInput.text("Say this is a test.").build();
                GeminiOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
                Assert.assertEquals(ret.getTextOutput(), "This is a test.");
            }
        }
    }

    @Test
    public void testStreamGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedStreamContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            server.setContentType("text/event-stream");

            String endpoint = "http://localhost:" + server.getPort();
            // endpoint = "https://generativelanguage.googleapis.com";
            String url = endpoint + "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse";

            Criteria<GeminiInput, StreamGeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, StreamGeminiOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234")
                            .build();

            try (ZooModel<GeminiInput, StreamGeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, StreamGeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in = GeminiInput.text("Say this is a test.").build();
                StreamGeminiOutput ret = predictor.predict(in);
                StringBuilder sb = new StringBuilder();
                for (GeminiOutput out : ret) {
                    logger.info(out.getTextOutput());
                    sb.append(out.getTextOutput());
                }
                Assert.assertEquals(sb.toString(), "This is a test.");
            }
        }
    }

    @Test
    public void testImageUnderstand() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String endpoint = "http://localhost:" + server.getPort();
            // endpoint = "https://generativelanguage.googleapis.com";
            String url = endpoint + "/v1beta/models/gemini-2.5-flash:generateContent";

            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234")
                            .build();
            Path testImage = Paths.get("../../examples/src/test/resources/kitten.jpg");
            byte[] bytes = Files.readAllBytes(testImage);

            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in =
                        GeminiInput.bytes(bytes, "image/jpeg")
                                .addText("Caption this image.")
                                .build();
                GeminiOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
                Assert.assertEquals(ret.getTextOutput(), "This is a test.");
            }
        }
    }

    @Test
    public void testFileUri() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String endpoint = "http://localhost:" + server.getPort();
            // endpoint = "https://aiplatform.googleapis.com";
            // file uri only requires Vertex URL
            String url =
                    endpoint
                            + "/v1/projects/$PROJECT/locations/global/publishers/google/models/gemini-2.5-flash:generateContent";

            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234")
                            .build();
            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in =
                        GeminiInput.fileUri(
                                        "gs://cloud-samples-data/generative-ai/image/scones.jpg",
                                        "image/jpeg")
                                .addText("Caption this image.")
                                .build();
                GeminiOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
                Assert.assertEquals(ret.getTextOutput(), "This is a test.");
            }
        }
    }

    @Test
    public void testGenerateContentWithConfig()
            throws ModelException, IOException, TranslateException {
        GenerationConfig config =
                GenerationConfig.builder()
                        .candidateCount(1)
                        .maxOutputTokens(1024)
                        .systemInstruction("You are a history teacher.")
                        .addSafetySetting(
                                SafetySetting.builder()
                                        .category(HarmCategory.HARM_CATEGORY_HATE_SPEECH)
                                        .threshold(HarmBlockThreshold.BLOCK_ONLY_HIGH))
                        .addTool(Tool.builder().googleSearch(GoogleSearch.builder()))
                        .build();

        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String endpoint = "http://localhost:" + server.getPort();
            // endpoint = "https://generativelanguage.googleapis.com";
            String url = endpoint + "/v1beta/models/gemini-2.5-flash:generateContent";

            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234")
                            .build();

            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in = GeminiInput.text("Tell me the history of LLM", config).build();
                GeminiOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
            }
        }
    }

    private String loadGeneratedContent() {
        return "{\"candidates\": [{\"content\":{\"parts\":[{\"text\":\"This is a test.\"}],"
                + "\"role\": \"model\"},\"finishReason\": \"STOP\",\"index\":0}]}";
    }

    private String loadGeneratedStreamContent() {
        return "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"This\"}],"
                + "\"role\": \"model\"},\"finishReason\": \"STOP\",\"index\":0}]}\n\n"
                + "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" is a test.\"}],"
                + "\"role\": \"model\"},\"finishReason\": \"STOP\",\"index\":0}]}\n\n";
    }
}
