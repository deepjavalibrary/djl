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
package ai.djl.genai.huggingface;

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

public class TextGenerationTest {

    private static final Logger logger = LoggerFactory.getLogger(TextGenerationTest.class);

    @Test
    public void testGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            String endpoint = "http://localhost:" + server.getPort();
            String url = endpoint + "/generate";
            Criteria<GenerationInput, GenerationOutput[]> criteria =
                    Criteria.builder()
                            .setTypes(GenerationInput.class, GenerationOutput[].class)
                            .optModelUrls(url)
                            .build();

            try (ZooModel<GenerationInput, GenerationOutput[]> model = criteria.loadModel();
                    Predictor<GenerationInput, GenerationOutput[]> predictor =
                            model.newPredictor()) {
                GenerationInput in =
                        GenerationInput.text("Say this is a test.")
                                .config(GenerationConfig.builder().maxNewTokens(20))
                                .build();
                GenerationOutput[] ret = predictor.predict(in);
                Assert.assertTrue(ret.length > 0);
                String text = ret[0].getGeneratedText();
                logger.info(text);
                Assert.assertEquals(text, "This is a test.");
                Details details = ret[0].getDetails();
                Assert.assertEquals(details.getFinishReason(), "length");
                Assert.assertEquals(details.getGeneratedTokens(), 128);
            }
        }
    }

    @Test
    public void testStreamGenerateContent() throws ModelException, IOException, TranslateException {
        String mockResponse = loadGeneratedStreamContent();
        try (TestServer server = TestServer.newInstance(mockResponse)) {
            server.setContentType("text/event-stream");

            String endpoint = "http://localhost:" + server.getPort();
            String url = endpoint + "/generate";

            Criteria<GenerationInput, StreamGenerationOutput> criteria =
                    Criteria.builder()
                            .setTypes(GenerationInput.class, StreamGenerationOutput.class)
                            .optModelUrls(url)
                            .build();

            try (ZooModel<GenerationInput, StreamGenerationOutput> model = criteria.loadModel();
                    Predictor<GenerationInput, StreamGenerationOutput> predictor =
                            model.newPredictor()) {
                GenerationInput in =
                        GenerationInput.text("Say this is a test.").stream(true).build();
                StreamGenerationOutput ret = predictor.predict(in);
                StringBuilder sb = new StringBuilder();
                for (GenerationOutput out : ret) {
                    Token token = out.getToken();
                    if (token != null && token.getText() != null) {
                        logger.info(token.getText());
                        sb.append(token.getText());
                    }
                }
                Assert.assertEquals(sb.toString(), "This is a test.");
            }
        }
    }

    private String loadGeneratedContent() {
        return "[{\"generated_text\": \"This is a test.\",\"details\": {\n"
                + "   \"finish_reason\": \"length\",\"generated_tokens\": 128,\"inputs\": \"Say"
                + " this is a test.\",\n"
                + "   \"tokens\": [{\"token\": {\"id\": 1, \"text\": \"This\", \"log_prob\":"
                + " -3.9}},{\"token\": {\"id\": 2, \"text\": \" is\", \"log_prob\": -2.1}}],\n"
                + "   \"prefill\": [{\"token\": {\"id\": 3,\"text\": \" test\", \"log_prob\":"
                + " -0.1}}]}}]";
    }

    private String loadGeneratedStreamContent() {
        return "data: {\"token\":{\"id\":1,\"text\":\"This\",\"log_prob\":-0.05}}\n\n"
                + "data: {\"token\":{\"id\":2,\"text\":\" is\",\"log_prob\":-0.05}}\n\n"
                + "data: {\"token\":{\"id\":3,\"text\":\" a\",\"log_prob\":-0.05}}\n\n"
                + "data: {\"token\":{\"id\":4,\"text\":\" test.\",\"log_prob\":-0.05}}\n\n";
    }
}
