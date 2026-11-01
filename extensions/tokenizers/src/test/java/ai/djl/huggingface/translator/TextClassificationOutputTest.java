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
package ai.djl.huggingface.translator;

import ai.djl.Model;
import ai.djl.inference.Predictor.PredictorContext;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class TextClassificationOutputTest {

    @DataProvider(name = "classificationOutputs")
    public Object[][] classificationOutputs() {
        return new Object[][] {
            {"Rust", null, new float[] {0.2f, 1.3f, -0.7f}},
            {"Rust", "single_label_classification", new float[] {-4f, -2f, -3f}},
            {"Rust", "multi_label_classification", new float[] {-2f, 3f, 0f}},
            {"Rust", null, new float[] {0.7f}},
            {"Rust", "multi_label_classification", new float[] {2f, 2f, -1f}},
            {"Rust", null, new float[] {0f, 0f, 0f}},
            {"PyTorch", null, new float[] {0.2f, 1.3f, -0.7f}},
            {"PyTorch", "single_label_classification", new float[] {-4f, -2f, -3f}},
            {"PyTorch", "multi_label_classification", new float[] {-2f, 3f, 0f}},
            {"PyTorch", null, new float[] {0.7f}},
            {"PyTorch", "multi_label_classification", new float[] {2f, 2f, -1f}},
            {"PyTorch", null, new float[] {0f, 0f, 0f}}
        };
    }

    @Test(dataProvider = "classificationOutputs")
    public void testOutput(String engine, String problemType, float[] logits) throws Exception {
        Path directory = createConfig(problemType, logits.length);
        try (NDManager manager = NDManager.newBaseManager(engine);
                TranslatorContext context = createContext(directory, manager)) {
            TextClassificationTranslator translator =
                    TextClassificationTranslator.builder(null).build();
            translator.prepare(context);
            Classifications result =
                    translator.processOutput(context, new NDList(manager.create(logits)));
            assertOutput(result, logits, problemType);
        } finally {
            Files.delete(directory.resolve("config.json"));
            Files.delete(directory);
        }
    }

    @DataProvider(name = "engines")
    public Object[][] engines() {
        return new Object[][] {{"Rust"}, {"PyTorch"}};
    }

    @Test(dataProvider = "engines")
    public void testBatchOutput(String engine) throws Exception {
        float[][] logits = {{0.2f, 1.3f, -0.7f}, {2f, -1f, 0f}};
        Path directory = createConfig(null, 3);
        try (NDManager manager = NDManager.newBaseManager(engine);
                TranslatorContext context = createContext(directory, manager)) {
            TextClassificationTranslator translator =
                    TextClassificationTranslator.builder(null).build();
            translator.prepare(context);
            List<Classifications> results =
                    translator.batchProcessOutput(context, new NDList(manager.create(logits)));
            Assert.assertEquals(results.size(), logits.length);
            for (int i = 0; i < logits.length; ++i) {
                assertOutput(results.get(i), logits[i], null);
            }
        } finally {
            Files.delete(directory.resolve("config.json"));
            Files.delete(directory);
        }
    }

    private static Path createConfig(String problemType, int size) throws Exception {
        Files.createDirectories(Paths.get("build"));
        Path directory = Files.createTempDirectory(Paths.get("build"), "classification-output-");
        Map<String, Object> config = new HashMap<>();
        Map<String, String> labels = new HashMap<>();
        for (int i = 0; i < size; ++i) {
            labels.put(String.valueOf(i), "LABEL_" + i);
        }
        config.put("id2label", labels);
        config.put("problem_type", problemType);
        Files.writeString(directory.resolve("config.json"), JsonUtils.GSON.toJson(config));
        return directory;
    }

    private static TranslatorContext createContext(Path directory, NDManager manager) {
        // Only model metadata is stubbed; arrays and all output operations use the real engine.
        Model model =
                (Model)
                        Proxy.newProxyInstance(
                                Model.class.getClassLoader(),
                                new Class<?>[] {Model.class},
                                (proxy, method, args) -> {
                                    if ("getModelPath".equals(method.getName())) {
                                        return directory;
                                    }
                                    throw new UnsupportedOperationException(method.getName());
                                });
        return new PredictorContext(model, manager, null);
    }

    private static void assertOutput(Classifications result, float[] logits, String problemType) {
        List<String> labels = result.getClassNames();
        List<Double> probabilities = result.getProbabilities();
        Assert.assertEquals(labels.size(), logits.length);
        Assert.assertEquals(probabilities.size(), logits.length);
        Assert.assertEquals(new HashSet<>(labels).size(), logits.length);
        double sum = 0;
        for (float logit : logits) {
            sum += Math.exp(logit);
        }
        for (int i = 0; i < logits.length; ++i) {
            int index = Integer.parseInt(labels.get(i).substring("LABEL_".length()));
            double expected =
                    "multi_label_classification".equals(problemType) || logits.length == 1
                            ? 1 / (1 + Math.exp(-logits[index]))
                            : Math.exp(logits[index]) / sum;
            Assert.assertEquals(probabilities.get(i), expected, 1e-6);
            if (i > 0) {
                Assert.assertTrue(probabilities.get(i - 1) >= probabilities.get(i));
            }
        }
        Assert.assertEquals(result.best().getProbability(), probabilities.get(0));
    }
}
