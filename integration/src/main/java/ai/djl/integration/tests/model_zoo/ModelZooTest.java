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
import ai.djl.ndarray.NDList;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.util.Utils;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Calendar;
import java.util.List;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class ModelZooTest {

    @BeforeClass
    public void setUp() {
        // force downloading without cache in .djl.ai folder.
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        String userHome = System.getProperty("user.home");
        System.setProperty("ENGINE_CACHE_DIR", userHome + "/.djl.ai");
    }

    @AfterClass
    public void tearDown() {
        System.clearProperty("DJL_CACHE_DIR");
        System.clearProperty("ENGINE_CACHE_DIR");
    }

    @Test
    public void testDownloadModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            throw new SkipException("Weekly only");
        }
        if (Calendar.SATURDAY != Calendar.getInstance().get(Calendar.DAY_OF_WEEK)) {
            throw new SkipException("Weekly only");
        }

        for (ModelZoo zoo : ModelZoo.listModelZoo()) {
            for (ModelLoader modelLoader : zoo.getModelLoaders()) {
                List<Artifact> artifacts = modelLoader.listModels();
                for (Artifact artifact : artifacts) {
                    Criteria<NDList, NDList> criteria =
                            Criteria.builder()
                                    .setTypes(NDList.class, NDList.class)
                                    .optFilters(artifact.getProperties())
                                    .build();
                    Model model = modelLoader.loadModel(criteria);
                    model.close();
                }
                Utils.deleteQuietly(Paths.get("build/cache"));
            }
        }
    }
}
