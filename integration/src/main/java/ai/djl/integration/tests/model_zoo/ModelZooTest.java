/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.integration.tests.model_zoo;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import java.io.IOException;
import java.util.List;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class ModelZooTest {

    @BeforeClass
    public void setUp() {
        // force downloading without cache in .djl.ai folder.
        System.setProperty("DJL_CACHE_DIR", "build/cache");
    }

    @AfterClass
    public void tearDown() {
        System.setProperty("DJL_CACHE_DIR", "");
    }

    @Test
    public void testDownloadModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        List<ModelLoader<?, ?>> list = ModelZoo.getModelZoo(BasicModelZoo.NAME).getModelLoaders();
        for (ModelLoader<?, ?> modelLoader : list) {
            List<Artifact> artifacts = modelLoader.listModels();
            for (Artifact artifact : artifacts) {
                Model model = modelLoader.loadModel(artifact.getProperties());
                model.close();
            }
        }
    }
}
