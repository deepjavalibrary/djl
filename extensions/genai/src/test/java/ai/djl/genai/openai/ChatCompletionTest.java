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
package ai.djl.genai.openai;

import ai.djl.ModelException;
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

public class ChatCompletionTest {

    private static final Logger logger = LoggerFactory.getLogger(ChatCompletionTest.class);

    @Test
    public void testGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String endpoint = "http://localhost:" + server.getPort();
            // endpoint = "https://generativelanguage.googleapis.com";
            String url = endpoint + "/v1beta/openai/chat/completions";
            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234") // override env var
                            .build();

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.text("Say this is a test.").model("gemini-2.5-flash").build();
                ChatOutput ret = predictor.predict(in);
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
            String url = endpoint + "/v1beta/openai/chat/completions";

            Criteria<ChatInput, StreamChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, StreamChatOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234")
                            .build();

            try (ZooModel<ChatInput, StreamChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, StreamChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.text("Say this is a test.").model("gemini-2.5-flash").stream(true)
                                .build();
                StreamChatOutput ret = predictor.predict(in);
                StringBuilder sb = new StringBuilder();
                for (ChatOutput out : ret) {
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
            String url = endpoint + "/v1beta/openai/chat/completions";

            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234") // override env var
                            .build();
            Path testImage = Paths.get("../../examples/src/test/resources/kitten.jpg");
            byte[] bytes = Files.readAllBytes(testImage);

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.file("1", bytes, "test.jpg")
                                .addText("Caption this image.")
                                .model("gemini-2.5-flash")
                                .build();
                ChatOutput ret = predictor.predict(in);
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
            String url = endpoint + "/v1beta/openai/chat/completions";

            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234") // override env var
                            .build();
            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.image("https://resources.djl.ai/images/kitten.jpg")
                                .addText("Caption this image.")
                                .model("gemini-2.5-flash")
                                .build();
                ChatOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
                Assert.assertEquals(ret.getTextOutput(), "This is a test.");
            }
        }
    }

    @Test
    public void testGenerateContentWithConfig()
            throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String endpoint = "http://localhost:" + server.getPort();
            // endpoint = "https://generativelanguage.googleapis.com";
            String url = endpoint + "/v1beta/openai/chat/completions";

            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(url)
                            .optArgument("API_KEY", "1234")
                            .build();

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.text("Tell me the history of LLM")
                                .model("gemini-2.5-flash")
                                .maxCompletionTokens(2000)
                                .build();
                ChatOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
                Assert.assertEquals(ret.getTextOutput(), "This is a test.");
            }
        }
    }

    private String loadGeneratedContent() {
        return "{\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"message\":"
                + "{\"content\":\"This is a test.\",\"role\":\"assistant\"}}],"
                + "\"created\":1751755529,\"id\":\"1\",\"model\":\"gemini-2.5-flash\","
                + "\"object\":\"chat.completion\",\"usage\":{\"completion_tokens\":5,"
                + "\"prompt_tokens\":7,\"total_tokens\":49}}";
    }

    private String loadGeneratedStreamContent() {
        return "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"message\":"
                + "{\"content\":\"This is\",\"role\":\"assistant\"}}],"
                + "\"created\":1751755529,\"id\":\"1\",\"model\":\"gemini-2.5-flash\","
                + "\"object\":\"chat.completion\",\"usage\":{\"completion_tokens\":5,"
                + "\"prompt_tokens\":7,\"total_tokens\":49}}\n\n"
                + "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"message\":"
                + "{\"content\":\" a test.\",\"role\":\"assistant\"}}],"
                + "\"created\":1751755529,\"id\":\"1\",\"model\":\"gemini-2.5-flash\","
                + "\"object\":\"chat.completion\",\"usage\":{\"completion_tokens\":5,"
                + "\"prompt_tokens\":7,\"total_tokens\":49}}\n\n";
    }
}
