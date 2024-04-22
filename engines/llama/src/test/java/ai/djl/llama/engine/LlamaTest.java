/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.llama.engine;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.engine.StandardCapabilities;
import ai.djl.inference.Predictor;
import ai.djl.llama.jni.Token;
import ai.djl.llama.jni.TokenIterator;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestRequirements;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;

public class LlamaTest {

    private static final Logger logger = LoggerFactory.getLogger(LlamaTest.class);

    @BeforeClass
    public void setUp() {
        System.setProperty("DJL_CACHE_DIR", "build/cache");
    }

    @AfterClass
    public void tierDown() {
        System.clearProperty("DJL_CACHE_DIR");
    }

    @Test
    public void testLlamaVersion() {
        Engine engine = Engine.getEngine("Llama");
        Assert.assertEquals(engine.getVersion(), "b1696-" + Engine.getDjlVersion());
        Assert.assertNotNull(engine.toString());
        Assert.assertEquals(engine.getRank(), 10);
        Assert.assertFalse(engine.hasCapability(StandardCapabilities.CUDA));
        Assert.assertNull(engine.getAlternativeEngine());
        try (NDManager manager = engine.newBaseManager()) {
            Assert.assertNotNull(manager);
        }
    }

    @Test
    public void testLlama() throws TranslateException, ModelException, IOException {
        TestRequirements.nightly();
        downloadModel();
        Path path = Paths.get("models");
        Criteria<String, TokenIterator> criteria =
                Criteria.builder()
                        .setTypes(String.class, TokenIterator.class)
                        .optModelPath(path)
                        .optModelName("tinyllama-1.1b-1t-openorca.Q4_K_M")
                        .optEngine("Llama")
                        .optOption("number_gpu_layers", "43")
                        .optTranslatorFactory(new LlamaTranslatorFactory())
                        .build();

        String prompt =
                "{\"inputs\": \"<|im_start|>system\n"
                        + "{system_message}<|im_end|>\n"
                        + "<|im_start|>user\n"
                        + "{prompt}<|im_end|>\n"
                        + "<|im_start|>assistant\", \"parameters\": {\"max_new_tokens\": 10}}";
        try (ZooModel<String, TokenIterator> model = criteria.loadModel();
                Predictor<String, TokenIterator> predictor = model.newPredictor()) {
            TokenIterator it = predictor.predict(prompt);
            StringBuilder sb = new StringBuilder();
            while (it.hasNext()) {
                Token token = it.next();
                Assert.assertNotNull(token.getText());
                Assert.assertTrue(token.getToken() >= 0);
                Assert.assertNotNull(token.getProbabilities());
                sb.append(token.getText());
                logger.info("{}", token);
            }
            Assert.assertTrue(sb.length() > 1);
        }
    }

    @Test
    public void testLlamaInfill() throws TranslateException, ModelException, IOException {
        TestRequirements.nightly();
        downloadModel();
        Path path = Paths.get("models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf");
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optOption("number_gpu_layers", "43")
                        .optEngine("Llama")
                        .optTranslatorFactory(new LlamaTranslatorFactory())
                        .build();

        String prompt =
                "{\n"
                        + "   \"prefix\":\"def remove_non_ascii(s: str) -> str:\n\",\n"
                        + "   \"suffix\":\"\n    return result\n\",\n"
                        + "   \"parameters\":{\n"
                        + "      \"max_new_tokens\": 10"
                        + "   }\n"
                        + "}";
        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input in = new Input();
            in.add("data", prompt);
            Output out = predictor.predict(in);
            Assert.assertNotNull(out.getData().getAsString());
        }
    }

    private void downloadModel() throws IOException {
        String url =
                "https://resources.djl.ai/test-models/gguf/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf";
        Path dir = Paths.get("models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf");
        DownloadUtils.download(URI.create(url).toURL(), dir, null);
    }
}
