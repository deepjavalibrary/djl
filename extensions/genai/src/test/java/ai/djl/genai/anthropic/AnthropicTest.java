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
package ai.djl.genai.anthropic;

import ai.djl.ModelException;
import ai.djl.genai.FunctionUtils;
import ai.djl.genai.openai.ChatInput;
import ai.djl.genai.openai.ChatOutput;
import ai.djl.genai.openai.Function;
import ai.djl.genai.openai.StreamChatOutput;
import ai.djl.genai.openai.Tool;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestServer;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class AnthropicTest {

    private static final Logger logger = LoggerFactory.getLogger(AnthropicTest.class);

    private String baseUrl;

    private void setMockBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    @BeforeClass
    public void setUp() {
        System.setProperty("PROJECT", "MY_FIRST_PROJECT");
    }

    @AfterClass
    public void tierDown() {
        System.clearProperty("PROJECT");
    }

    @Test
    public void testGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(Anthropic.SONNET_4.getUrl(baseUrl))
                            .build();

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.text("Say this is a test.")
                                .inputType(ChatInput.Type.ANTHROPIC_VERTEX)
                                .build();
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
            setMockBaseUrl("http://localhost:" + server.getPort());

            Criteria<ChatInput, StreamChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, StreamChatOutput.class)
                            .optModelUrls(Anthropic.SONNET_4.getUrl(baseUrl, true))
                            .build();
            try (ZooModel<ChatInput, StreamChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, StreamChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.text("Say this is a test.").stream(true)
                                .inputType(ChatInput.Type.ANTHROPIC_VERTEX)
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
    public void testGenerateContentWithFunction()
            throws ModelException, IOException, TranslateException, ReflectiveOperationException {
        String mockResponse = loadFunctionContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(Anthropic.SONNET_4.getUrl(baseUrl))
                            .build();

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                Method method = AnthropicTest.class.getMethod("getWeather", String.class);
                Function function =
                        Function.function(method)
                                .description("Get the current weather in a given location")
                                .build();
                ChatInput in =
                        ChatInput.text("What's the weather like in New York today?")
                                .inputType(ChatInput.Type.ANTHROPIC_VERTEX)
                                .tools(Tool.of(function))
                                .toolChoice("auto")
                                .build();

                ChatOutput ret = predictor.predict(in);
                String arguments = ret.getToolCall().getFunction().getArguments();
                String weather = (String) FunctionUtils.invoke(method, this, arguments);
                Assert.assertEquals(weather, "nice");
            }
        }
    }

    @Test
    public void testImageUnderstand() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(Anthropic.SONNET_4.getUrl(baseUrl))
                            .build();
            Path testImage = Paths.get("../../examples/src/test/resources/kitten.jpg");
            byte[] bytes = Files.readAllBytes(testImage);

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.image(bytes, "image/jpeg")
                                .addText("Caption this image.")
                                .inputType(ChatInput.Type.ANTHROPIC_VERTEX)
                                .build();
                ChatOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
                Assert.assertEquals(ret.getTextOutput(), "This is a test.");
            }
        }
    }

    private String loadGeneratedContent() {
        return "{\"id\":\"1\",\"type\":\"message\","
                + "\"role\":\"assistant\",\"model\":\"claude-3-5-haiku-20241022\","
                + "\"content\":[{\"type\":\"text\",\"text\":\"This is a test.\"}],"
                + "\"stop_reason\":\"end_turn\",\"stop_sequence\":null,\"usage\":"
                + "{\"input_tokens\":13,\"cache_creation_input_tokens\":0,"
                + "\"cache_read_input_tokens\":0,\"output_tokens\":8}}";
    }

    private String loadGeneratedStreamContent() {
        return "event: message_start\n"
                   + "data:"
                   + " {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-haiku-20241022\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":13,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":1}}}\n\n"
                   + "event: ping\n"
                   + "data: {\"type\": \"ping\"}\n\n"
                   + "event: content_block_start\n"
                   + "data:"
                   + " {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}"
                   + " }\n\n"
                   + "event: content_block_delta\n"
                   + "data:"
                   + " {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"This\"}}\n\n"
                   + "event: content_block_delta\n"
                   + "data:"
                   + " {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\""
                   + " is a test.\"}}\n\n"
                   + "event: content_block_stop\n"
                   + "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
                   + "event: message_delta\n"
                   + "data:"
                   + " {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":8}}\n\n"
                   + "event: message_stop\n"
                   + "data: {\"type\":\"message_stop\"}\n\n";
    }

    private String loadFunctionContent() {
        return "{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\","
                   + "\"model\":\"claude-sonnet-4-20250514\",\"content\":[{\"type\":\"text\",\"text\":\"I'll"
                   + " check the current weather in New York for"
                   + " you.\"},{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"getWeather\",\"input\":{\"location\":\"New"
                   + " York\"}}],"
                   + "\"stop_reason\":\"tool_use\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":386,"
                   + "\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":67}}";
    }

    public String getWeather(String location) {
        return "nice";
    }
}
