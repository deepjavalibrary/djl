/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference;

import ai.djl.Application;
import ai.djl.Application.CV;
import ai.djl.Application.NLP;
import ai.djl.ModelException;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

public class ListModelsTest {

    @AfterClass
    public void tearDown() {
        System.clearProperty("ai.djl.repository.zoo.location");
    }

    @Test
    public void testListModels() throws ModelException, IOException {
        Path path = Paths.get("../model-zoo/src/test/resources/mlrepo");
        String repoUrl = path.toRealPath().toAbsolutePath().toUri().toURL().toExternalForm();
        System.setProperty("ai.djl.repository.zoo.location", "src/test/resources," + repoUrl);
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        System.out.println(Arrays.toString(models.keySet().toArray()));
        List<Artifact> artifacts = models.get(Application.UNDEFINED);
        Assert.assertFalse(artifacts.isEmpty());
    }

    @Test
    public void testListModelsWithApplication() throws ModelException, IOException {
        Path path = Paths.get("../model-zoo/src/test/resources/mlrepo");
        String repoUrl = path.toRealPath().toAbsolutePath().toUri().toURL().toExternalForm();
        System.setProperty("ai.djl.repository.zoo.location", "src/test/resources," + repoUrl);
        Criteria<?, ?> criteria = Criteria.builder().optApplication(NLP.ANY).build();
        Map<Application, List<Artifact>> models = ModelZoo.listModels(criteria);

        for (Application application : models.keySet()) {
            Assert.assertTrue(
                    application.matches(NLP.ANY) || application.matches(Application.UNDEFINED));
            Assert.assertFalse(application.matches(CV.ANY));
        }
    }
}
