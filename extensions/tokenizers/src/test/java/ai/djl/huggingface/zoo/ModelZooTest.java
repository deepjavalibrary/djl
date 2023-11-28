/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.huggingface.zoo;

import ai.djl.Application.NLP;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class ModelZooTest {

    @Test
    public void testModelZoo() throws ModelException, IOException, TranslateException {
        TestRequirements.nightly();

        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        String url = "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2";
        Criteria<QAInput, String> criteria =
                Criteria.builder().setTypes(QAInput.class, String.class).optModelUrls(url).build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            QAInput input = new QAInput(question, paragraph);
            String res = predictor.predict(input);
            Assert.assertEquals(res, "december 2004");
        }
    }

    @Test
    public void testFutureVersion() throws IOException {
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        try {
            Utils.deleteQuietly(Paths.get("build/cache"));
            Map<String, Map<String, Object>> map = new ConcurrentHashMap<>();
            Map<String, Object> model = new ConcurrentHashMap<>();
            model.put("result", "failed");
            map.put("model1", model);

            model = new ConcurrentHashMap<>();
            model.put("requires", "10.100.0+");
            map.put("model2", model);

            model = new ConcurrentHashMap<>();
            model.put("requires", "0.19.0+");
            map.put("model3", model);
            map.put("model4", new ConcurrentHashMap<>());

            String path = "model/" + NLP.QUESTION_ANSWER.getPath() + "/ai/djl/huggingface/pytorch/";
            Path dir = Utils.getCacheDir().resolve("cache/repo/" + path);
            Files.createDirectories(dir);
            Path file = dir.resolve("models.json");
            try (Writer writer = Files.newBufferedWriter(file)) {
                writer.write(JsonUtils.GSON_PRETTY.toJson(map));
            }

            // static variables cannot not be initialized properly if directly use new HfModelZoo()
            ModelZoo.getModelZoo("ai.djl.huggingface.pytorch");
            ModelZoo zoo = new HfModelZoo();

            Assert.assertNull(zoo.getModelLoader("model1"));
            Assert.assertNull(zoo.getModelLoader("model2"));
            Assert.assertNull(zoo.getModelLoader("model3"));
            Assert.assertNotNull(zoo.getModelLoader("model4"));
        } finally {
            System.clearProperty("DJL_CACHE_DIR");
        }
    }

    @Test
    public void testOffLine() throws IOException {
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        System.setProperty("ai.djl.offline", "true");
        try {
            Utils.deleteQuietly(Paths.get("build/cache"));
            // static variables cannot not be initialized properly if directly use new HfModelZoo()
            ModelZoo.getModelZoo("ai.djl.huggingface.pytorch");

            ModelZoo zoo = new HfModelZoo();
            Assert.assertFalse(zoo.getModelLoaders().isEmpty());

            Set<String> engines = zoo.getSupportedEngines();
            Assert.assertEquals(engines.size(), 1);
            Assert.assertEquals(engines.iterator().next(), "PyTorch");
        } finally {
            System.clearProperty("DJL_CACHE_DIR");
            System.clearProperty("ai.djl.offline");
        }
    }
}
