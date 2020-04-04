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
import ai.djl.ModelException;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelZoo;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ListModelsTest {

    @Test
    public void testListModels() throws ModelException, IOException {
        Path path = Paths.get("../model-zoo/src/test/resources/mlrepo");
        String repoUrl = path.toRealPath().toAbsolutePath().toUri().toURL().toExternalForm();
        System.setProperty("ai.djl.repository.zoo.location", "src/test/resources," + repoUrl);
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        List<Artifact> artifacts = models.get(Application.UNDEFINED);
        Assert.assertTrue(artifacts.size() > 1);
    }
}
