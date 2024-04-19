/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.onnxruntime.zoo;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Set;

public class OrtHfModelZooTest {

    @Test
    public void testHfModelZoo() throws ModelException, IOException, TranslateException {
        TestRequirements.nightly();

        String text = "What is deep learning?";
        String url = "djl://ai.djl.huggingface.onnxruntime/TaylorAI/bge-micro-v2";
        Criteria<String, float[]> criteria =
                Criteria.builder().setTypes(String.class, float[].class).optModelUrls(url).build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
                Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 384);
        }
    }

    @Test
    public void testOffLine() {
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        System.setProperty("ai.djl.offline", "true");
        try {
            Utils.deleteQuietly(Paths.get("build/cache"));
            // static variables cannot not be initialized properly if directly use new HfModelZoo()
            ModelZoo.getModelZoo("ai.djl.huggingface.onnxruntime");

            ModelZoo zoo = new OrtHfModelZoo();
            Assert.assertFalse(zoo.getModelLoaders().isEmpty());

            Set<String> engines = zoo.getSupportedEngines();
            Assert.assertEquals(engines.size(), 1);
            Assert.assertEquals(engines.iterator().next(), "OnnxRuntime");
        } finally {
            System.clearProperty("DJL_CACHE_DIR");
            System.clearProperty("ai.djl.offline");
        }
    }
}
