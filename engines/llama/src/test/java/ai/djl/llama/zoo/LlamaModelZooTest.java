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
package ai.djl.llama.zoo;

import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.file.Paths;
import java.util.Collection;

public class LlamaModelZooTest {

    @Test
    public void testLlamaModelZoo() {
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        Utils.deleteQuietly(Paths.get("build/cache/cache"));
        try {
            ModelZoo zoo = ModelZoo.getModelZoo("ai.djl.huggingface.gguf");
            Collection<ModelLoader> models = zoo.getModelLoaders();
            Assert.assertFalse(models.isEmpty());
            Assert.assertEquals(zoo.getSupportedEngines().size(), 1);
            ModelLoader loader = zoo.getModelLoader("TinyLlama/TinyLlama-1.1B-Chat-v0.6");
            Assert.assertNotNull(loader);

            ModelZoo llamaModelZoo = new LlamaModelZoo();
            Assert.assertFalse(llamaModelZoo.getModelLoaders().isEmpty());
        } finally {
            System.clearProperty("DJL_CACHE_DIR");
        }
    }

    @Test
    public void testOffLine() {
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        System.setProperty("ai.djl.offline", "true");
        Utils.deleteQuietly(Paths.get("build/cache/cache"));
        try {
            // static variables cannot not be initialized properly if directly use LlamaModelZoo()
            ModelZoo.getModelZoo("ai.djl.huggingface.gguf");

            ModelZoo zoo = new LlamaModelZoo();
            Assert.assertFalse(zoo.getModelLoaders().isEmpty());
        } finally {
            System.clearProperty("DJL_CACHE_DIR");
            System.clearProperty("ai.djl.offline");
        }
    }
}
