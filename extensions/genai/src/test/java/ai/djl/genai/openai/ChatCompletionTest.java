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
import ai.djl.genai.FunctionUtils;
import ai.djl.genai.gemini.Gemini;
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
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class ChatCompletionTest {

    private static final Logger logger = LoggerFactory.getLogger(ChatCompletionTest.class);

    @Test
    public void testGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini =
                    Gemini.GEMINI_2_5_FLASH.toBuilder()
                            .baseUrl(baseUrl)
                            .chatCompletions(true)
                            .build();
            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(gemini.getUrl())
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

            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini =
                    Gemini.GEMINI_2_5_FLASH.toBuilder()
                            .baseUrl(baseUrl)
                            .chatCompletions(true)
                            .build();

            Criteria<ChatInput, StreamChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, StreamChatOutput.class)
                            .optModelUrls(gemini.getUrl())
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
    public void testGenerateContentWithFunction()
            throws ModelException, IOException, TranslateException, ReflectiveOperationException {
        String mockResponse = loadFunctionContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini =
                    Gemini.GEMINI_2_5_FLASH.toBuilder()
                            .baseUrl(baseUrl)
                            .chatCompletions(true)
                            .build();
            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(gemini.getUrl())
                            .optArgument("API_KEY", "1234") // override env var
                            .build();

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                Method method = ChatCompletionTest.class.getMethod("getWeather", String.class);
                Function function =
                        Function.function(method)
                                .description("Get the current weather in a given location")
                                .build();
                ChatInput in =
                        ChatInput.text("What's the weather like in New York today?")
                                .model("gemini-2.5-flash")
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
    public void testGeminiWithFunction()
            throws ModelException, IOException, TranslateException, ReflectiveOperationException {
        String mockResponse = loadGeminiFunctionContent();

        Method method = ChatCompletionTest.class.getMethod("getWeather", String.class);
        Function function =
                Function.function(method)
                        .description("Get the current weather in a given location")
                        .build();

        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini = Gemini.GEMINI_2_5_FLASH.toBuilder().baseUrl(baseUrl).build();
            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(gemini.getUrl())
                            .optArgument("API_KEY", "1234") // override env var
                            .build();

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {

                ChatInput in =
                        ChatInput.text("What is the weather like in celsius in New York today?")
                                .inputType(ChatInput.Type.GEMINI)
                                .tools(ai.djl.genai.openai.Tool.of(function))
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
            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini =
                    Gemini.GEMINI_2_5_FLASH.toBuilder()
                            .baseUrl(baseUrl)
                            .chatCompletions(true)
                            .build();

            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(gemini.getUrl())
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
            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini = Gemini.GEMINI_2_5_FLASH.toBuilder().baseUrl(baseUrl).build();

            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(gemini.getUrl())
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
            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini =
                    Gemini.GEMINI_2_5_FLASH.toBuilder()
                            .baseUrl(baseUrl)
                            .chatCompletions(true)
                            .build();

            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(gemini.getUrl())
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

    @Test
    public void testGeminiWithLogprobs() throws ModelException, IOException, TranslateException {
        String mockResponse = loadLogprobsContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String baseUrl = "http://localhost:" + server.getPort();
            Gemini gemini =
                    Gemini.GEMINI_2_5_FLASH.toBuilder()
                            .baseUrl(baseUrl)
                            .chatCompletions(true)
                            .build();

            Criteria<ChatInput, ChatOutput> criteria =
                    Criteria.builder()
                            .setTypes(ChatInput.class, ChatOutput.class)
                            .optModelUrls(gemini.getUrl())
                            .optArgument("API_KEY", "1234")
                            .build();

            try (ZooModel<ChatInput, ChatOutput> model = criteria.loadModel();
                    Predictor<ChatInput, ChatOutput> predictor = model.newPredictor()) {
                ChatInput in =
                        ChatInput.text("I am not sure if I really like this restaurant a lot.")
                                .inputType(ChatInput.Type.GEMINI)
                                .logprobs(true)
                                .topLogprobs(2)
                                .build();
                ChatOutput ret = predictor.predict(in);
                for (Logprob logprobs : ret.getLogprobs()) {
                    String token = logprobs.getToken();
                    float prob = logprobs.getLogprob();
                    logger.info("Token: {} ({})", token, String.format("%.03f", prob));
                    List<TopLogprob> alternatives = logprobs.getTopLogprobs();
                    if (alternatives != null) {
                        for (TopLogprob alt : alternatives) {
                            logger.info("Alternative {} ({})", alt.getToken(), alt.getLogprob());
                        }
                    }
                }
                Assert.assertEquals(ret.getTextOutput(), "\"Neutral\"");
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
        return "event: content_block_delta\n"
                + "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"message\":"
                + "{\"content\":\"This is\",\"role\":\"assistant\"}}],"
                + "\"created\":1751755529,\"id\":\"1\",\"model\":\"gemini-2.5-flash\","
                + "\"object\":\"chat.completion\",\"usage\":{\"completion_tokens\":5,"
                + "\"prompt_tokens\":7,\"total_tokens\":49}}\n\n"
                + "event: content_block_delta\n"
                + "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"message\":"
                + "{\"content\":\" a test.\",\"role\":\"assistant\"}}],"
                + "\"created\":1751755529,\"id\":\"1\",\"model\":\"gemini-2.5-flash\","
                + "\"object\":\"chat.completion\",\"usage\":{\"completion_tokens\":5,"
                + "\"prompt_tokens\":7,\"total_tokens\":49}}\n\n";
    }

    private String loadFunctionContent() {
        return "{\"id\":\"\",\"object\":\"chat.completion\",\"created\":1,\"choices\":["
                + "{\"index\":0,\"message\":{\"role\":\"assistant\",\"tool_calls\":["
                + "{\"id\":\"\",\"type\":\"function\",\"function\":{\"arguments\":\""
                + "{\\\"location\\\":\\\"New York\\\"}\",\"name\":\"getWeather\"}}]},"
                + "\"finish_reason\":\"tool_calls\"}],\"model\":\"gemini-2.5-flash\",\"usage\":"
                + "{\"completion_tokens\":15,\"prompt_tokens\":50,\"total_tokens\":127}}";
    }

    private String loadGeminiFunctionContent() {
        return "{\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{"
                + "\"args\":{\"unit\":true,\"location\":\"New York\"},\"name\":\"getWeather\"}}],"
                + "\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}]}";
    }

    private String loadLogprobsContent() {
        return "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":["
                   + " {\"text\":\"\\\"Neutral\\\"\"}]},\"finishReason\":\"STOP\",\"avgLogprobs\":-10.37,"
                   + " \"logprobsResult\":{\"topCandidates\":[{\"candidates\":["
                   + " {\"token\":\"H\",\"logProbability\":-0.19,\"tokenId\":236814},"
                   + " {\"token\":\"He\",\"logProbability\":-2.37,\"tokenId\":2209}]},"
                   + " {\"candidates\":[{\"token\":\"ere\",\"logProbability\":-0.13,\"tokenId\":627},"
                   + " {\"token\":\"e\",\"logProbability\":-2.30,\"tokenId\":236744}]},"
                   + " {\"candidates\":[{\"token\":\"is\",\"logProbability\":0,\"tokenId\":563},"
                   + " {\"token\":\"i\",\"logProbability\":-16.70,\"tokenId\":858}]},"
                   + " {\"candidates\":[{\"token\":\"the\",\"logProbability\":-3.57e-07,\"tokenId\":506},"
                   + " {\"token\":\"\",\"logProbability\":-14.70,\"tokenId\":236743}]},"
                   + " {\"candidates\":[{\"token\":\"JSON\",\"logProbability\":-1.19e-06,\"tokenId\":10434},"
                   + " {\"token\":\"\",\"logProbability\":-14.12,\"tokenId\":236743}]},"
                   + " {\"candidates\":[{\"token\":\"requested\",\"logProbability\":-0.41,\"tokenId\":15633},"
                   + " {\"token\":\"re\",\"logProbability\":-1.51,\"tokenId\":544}]},"
                   + " {\"candidates\":[{\"token\":\":\",\"logProbability\":0,\"tokenId\":236787},"
                   + " {\"token\":\"har\",\"logProbability\":-1.26e+30,\"tokenId\":3968}]},"
                   + " {\"candidates\":[{\"token\":\"\\n"
                   + "\",\"logProbability\":0,\"tokenId\":107},"
                   + " {\"token\":\"har\",\"logProbability\":-1.26e+30,\"tokenId\":3968}]},"
                   + " {\"candidates\":[{\"token\":\"```\",\"logProbability\":-0.00,\"tokenId\":2717},"
                   + " {\"token\":\"`\",\"logProbability\":-4.89,\"tokenId\":236929}]},"
                   + " {\"candidates\":[{\"token\":\"json\",\"logProbability\":0,\"tokenId\":3723},"
                   + " {\"token\":\"j\",\"logProbability\":-18.95,\"tokenId\":236804}]},"
                   + " {\"candidates\":[{\"token\":\"\\n"
                   + "\",\"logProbability\":0,\"tokenId\":107},"
                   + " {\"token\":\"har\",\"logProbability\":-1.26e+30,\"tokenId\":3968}]},"
                   + " {\"candidates\":[{\"token\":\"\\\"\",\"logProbability\":-1.19e-06,\"tokenId\":236775},"
                   + " {\"token\":\"\\\"\",\"logProbability\":-13.65,\"tokenId\":623}]},"
                   + " {\"candidates\":[{\"token\":\"Neutral\",\"logProbability\":-6.67e-06,\"tokenId\":20809},"
                   + " {\"token\":\"Negative\",\"logProbability\":-11.97,\"tokenId\":63702}]},"
                   + " {\"candidates\":[{\"token\":\"\\\"\",\"logProbability\":0,\"tokenId\":236775},"
                   + " {\"token\":\"har\",\"logProbability\":-1.26e+30,\"tokenId\":3968}]},"
                   + " {\"candidates\":[{\"token\":\"\\n"
                   + "\",\"logProbability\":-7.15e-07,\"tokenId\":107},"
                   + " {\"token\":\"\",\"logProbability\":-14.64,\"tokenId\":236743}]},"
                   + " {\"candidates\":[{\"token\":\"```\",\"logProbability\":-1.19e-07,\"tokenId\":2717},"
                   + " {\"token\":\"``\",\"logProbability\":-16.53,\"tokenId\":2629}]}],"
                   + " \"chosenCandidates\":[{\"token\":\"H\",\"logProbability\":-0.19,\"tokenId\":236814},"
                   + " {\"token\":\"ere\",\"logProbability\":-0.13,\"tokenId\":627},"
                   + " {\"token\":\"is\",\"logProbability\":0,\"tokenId\":563},"
                   + " {\"token\":\"the\",\"logProbability\":-3.57e-07,\"tokenId\":506},"
                   + " {\"token\":\"JSON\",\"logProbability\":-1.19e-06,\"tokenId\":10434},"
                   + " {\"token\":\"requested\",\"logProbability\":-0.41,\"tokenId\":15633},"
                   + " {\"token\":\":\",\"logProbability\":0,\"tokenId\":236787}, {\"token\":\"\\n"
                   + "\",\"logProbability\":0,\"tokenId\":107}, "
                   + " {\"token\":\"```\",\"logProbability\":-0.00,\"tokenId\":2717},"
                   + " {\"token\":\"json\",\"logProbability\":0,\"tokenId\":3723}, {\"token\":\"\\n"
                   + "\",\"logProbability\":0,\"tokenId\":107},"
                   + " {\"token\":\"\\\"\",\"logProbability\":-1.19e-06,\"tokenId\":236775},"
                   + " {\"token\":\"Neutral\",\"logProbability\":-6.67e-06,\"tokenId\":20809},"
                   + " {\"token\":\"\\\"\",\"logProbability\":0,\"tokenId\":236775},"
                   + " {\"token\":\"\\n"
                   + "\",\"logProbability\":-7.15e-07,\"tokenId\":107},"
                   + " {\"token\":\"```\",\"logProbability\":-1.19e-07,\"tokenId\":2717}]}}],"
                   + " \"usageMetadata\":{\"promptTokenCount\":17,\"candidatesTokenCount\":3,"
                   + " \"totalTokenCount\":115,\"trafficType\":\"ON_DEMAND\",\"promptTokensDetails\":"
                   + " [{\"modality\":\"TEXT\",\"tokenCount\":17}],\"candidatesTokensDetails\":"
                   + " [{\"modality\":\"TEXT\",\"tokenCount\":3}],\"thoughtsTokenCount\":95},"
                   + " \"modelVersion\":\"gemini-2.5-flash\",\"createTime\":\"2025-07-21T03:01:13.244502Z\","
                   + " \"responseId\":\"1\"}";
    }

    public String getWeather(String location) {
        return "nice";
    }
}
