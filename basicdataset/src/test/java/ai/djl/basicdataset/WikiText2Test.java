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
package ai.djl.basicdataset;

import ai.djl.basicdataset.nlp.WikiText2;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class WikiText2Test {
    @Test
    public void testWikiText2TrainRemote() throws IOException {
        WikiText2 trainingSet = WikiText2.builder().optUsage(Dataset.Usage.TRAIN).build();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }

    @Test
    public void testWikiText2TrainLocal() throws IOException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");

        WikiText2 trainingSet =
                WikiText2.builder().optRepository(repository).optUsage(Dataset.Usage.TRAIN).build();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }

    @Test
    public void testWikiText2TestLocal() throws IOException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");

        WikiText2 trainingSet =
                WikiText2.builder().optRepository(repository).optUsage(Dataset.Usage.TEST).build();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }

    @Test
    public void testWikiText2ValidationLocal() throws IOException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");

        WikiText2 trainingSet =
                WikiText2.builder()
                        .optRepository(repository)
                        .optUsage(Dataset.Usage.VALIDATION)
                        .build();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }
}
