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

import static ai.djl.genai.gemini.types.Type.STRING;

import ai.djl.ModelException;
import ai.djl.genai.FunctionUtils;
import ai.djl.genai.gemini.types.FunctionCall;
import ai.djl.genai.gemini.types.FunctionDeclaration;
import ai.djl.genai.gemini.types.GenerationConfig;
import ai.djl.genai.gemini.types.GoogleSearch;
import ai.djl.genai.gemini.types.HarmBlockThreshold;
import ai.djl.genai.gemini.types.HarmCategory;
import ai.djl.genai.gemini.types.LogprobsResultCandidate;
import ai.djl.genai.gemini.types.SafetySetting;
import ai.djl.genai.gemini.types.Schema;
import ai.djl.genai.gemini.types.Tool;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestServer;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

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
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class GeminiTest {

    private static final Logger logger = LoggerFactory.getLogger(GeminiTest.class);

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
            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(Gemini.GEMINI_2_5_FLASH.getUrl(baseUrl))
                            .build();

            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in = GeminiInput.text("Say this is a test.").build();
                GeminiOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
                Assert.assertEquals(ret.getTextOutput(), "This is a test.");
            }
            String receivedInput = server.setReceivedInput();
            Assert.assertEquals(
                    receivedInput,
                    "{\"contents\":[{\"parts\":[{\"text\":\"Say this is a"
                            + " test.\"}],\"role\":\"user\"}]}");
        }
    }

    @Test
    public void testStreamGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedStreamContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            server.setContentType("text/event-stream");
            setMockBaseUrl("http://localhost:" + server.getPort());

            Criteria<GeminiInput, StreamGeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, StreamGeminiOutput.class)
                            .optModelUrls(Gemini.GEMINI_2_5_FLASH.getUrl(baseUrl, true))
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
            String receivedInput = server.setReceivedInput();
            Assert.assertEquals(
                    receivedInput,
                    "{\"contents\":[{\"parts\":[{\"text\":\"Say this is a"
                            + " test.\"}],\"role\":\"user\"}]}");
        }
    }

    @Test
    public void testGenerateContentWithFunction()
            throws ModelException, IOException, TranslateException, ReflectiveOperationException {
        String mockResponse = loadFunctionContent();

        Method method = GeminiTest.class.getMethod("getWeather", String.class, boolean.class);
        FunctionDeclaration function =
                FunctionDeclaration.function(method)
                        .description(
                                "Get the current weather in a given location, set unit to true for"
                                        + " celsius")
                        .build();
        GenerationConfig config =
                GenerationConfig.builder()
                        .candidateCount(1)
                        .addTool(Tool.builder().addFunctionDeclaration(function))
                        .build();

        try (TestServer server = TestServer.newInstance(mockResponse)) {
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(Gemini.GEMINI_2_5_FLASH.getUrl(baseUrl))
                            .build();

            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {

                GeminiInput in =
                        GeminiInput.text("What is the weather like in celsius in New York today?")
                                .generationConfig(config)
                                .build();

                GeminiOutput ret = predictor.predict(in);
                FunctionCall functionCall = ret.getFunctionCall();
                Assert.assertNotNull(functionCall);
                Map<String, Object> arguments = functionCall.getArgs();
                String weather = (String) FunctionUtils.invoke(method, this, arguments);
                Assert.assertEquals(weather, "nice");
            }
            String receivedInput = server.setReceivedInput();
            Assert.assertEquals(receivedInput, getExpectedFunctionContent());
        }
    }

    @Test
    public void testImageUnderstand() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(Gemini.GEMINI_2_5_FLASH.getUrl(baseUrl))
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
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(Gemini.GEMINI_2_5_FLASH.getUrl(baseUrl, false))
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
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(Gemini.GEMINI_2_5_FLASH.getUrl(baseUrl))
                            .build();

            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in = GeminiInput.text("Tell me the history of LLM", config).build();
                GeminiOutput ret = predictor.predict(in);
                logger.info(ret.getTextOutput());
            }
        }
    }

    @Test
    public void testGenerateContentWithLogprobs()
            throws ModelException, IOException, TranslateException {
        GenerationConfig config =
                GenerationConfig.builder()
                        .candidateCount(1)
                        .maxOutputTokens(1024)
                        .logprobs(3)
                        .responseLogprobs(true)
                        .responseSchema(
                                Schema.builder()
                                        .type(STRING)
                                        .enumName(Arrays.asList("Positive", "Negative", "Neutral")))
                        .responseMimeType("application/json")
                        .build();

        String mockResponse = loadLogprobsContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            setMockBaseUrl("http://localhost:" + server.getPort());
            Criteria<GeminiInput, GeminiOutput> criteria =
                    Criteria.builder()
                            .setTypes(GeminiInput.class, GeminiOutput.class)
                            .optModelUrls(Gemini.GEMINI_2_5_FLASH.getUrl(baseUrl))
                            .build();

            try (ZooModel<GeminiInput, GeminiOutput> model = criteria.loadModel();
                    Predictor<GeminiInput, GeminiOutput> predictor = model.newPredictor()) {
                GeminiInput in =
                        GeminiInput.text(
                                        "I am not sure if I really like this restaurant a lot.",
                                        config)
                                .build();
                GeminiOutput ret = predictor.predict(in);
                for (Pair<LogprobsResultCandidate, List<LogprobsResultCandidate>> pair :
                        ret.getLogprobsResult()) {
                    LogprobsResultCandidate lr = pair.getKey();
                    List<LogprobsResultCandidate> alternatives = pair.getValue();
                    String token = lr.getToken();
                    float prob = lr.getLogProbability();
                    logger.info("Token: {} ({})", token, String.format("%.03f", prob));
                    if (!alternatives.isEmpty()) {
                        logger.info(
                                "Alternative tokens: {} ({})", token, String.format("%.03f", prob));
                        for (LogprobsResultCandidate alt : alternatives) {
                            logger.info("\t{} ({})", alt.getToken(), alt.getLogProbability());
                        }
                    }
                }
                Assert.assertEquals(ret.getTextOutput(), "\"Neutral\"");
            }
            String receivedInput = server.setReceivedInput();
            Assert.assertEquals(receivedInput, getExpectedLogprobInput());
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

    private String loadFunctionContent() {
        return "{\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{"
                + "\"args\":{\"unit\":true,\"location\":\"New York\"},\"name\":\"getWeather\"}}],"
                + "\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}]}";
    }

    private String getExpectedFunctionContent() {
        return "{\"contents\":[{\"parts\":[{\"text\":"
                + "\"What is the weather like in celsius in New York today?\"}],"
                + "\"role\":\"user\"}],\"generationConfig\":{\"candidateCount\":1},"
                + "\"safetySettings\":[],\"tools\":[{\"functionDeclarations\":"
                + "[{\"description\":\"Get the current weather in a given location,"
                + " set unit to true for celsius\",\"name\":\"getWeather\",\"parameters\":"
                + "{\"anyOf\":[],\"properties\":{\"unit\":{\"anyOf\":[],\"type\":\"BOOLEAN\"},"
                + "\"location\":{\"anyOf\":[],\"type\":\"STRING\"}},\"required\":[\"location\","
                + "\"unit\"],\"type\":\"OBJECT\"}}]}]}";
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

    private String getExpectedLogprobInput() {
        return "{\"contents\":[{\"parts\":[{\"text\":\"I am not sure if I really like this"
                   + " restaurant a lot.\"}],"
                   + "\"role\":\"user\"}],\"generationConfig\":{\"candidateCount\":1,\"logprobs\":3,"
                   + "\"maxOutputTokens\":1024,\"responseLogprobs\":true,\"responseMimeType\":\"application/json\","
                   + "\"responseSchema\":{\"anyOf\":[],\"enum\":[\"Positive\",\"Negative\",\"Neutral\"],"
                   + "\"type\":\"STRING\"}},\"safetySettings\":[],\"tools\":[]}";
    }

    public String getWeather(String location, boolean unit) {
        return "nice";
    }
}
