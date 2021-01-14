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
package ai.djl.basicdataset;

import ai.djl.basicdataset.nlp.CookingStackExchange;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CookingExchangeTest {

    @Test
    public void testTrainingDataset() throws IOException, TranslateException {
        CookingStackExchange trainingSet = CookingStackExchange.builder().build();
        trainingSet.prepare();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }

    @Test
    public void testTestDataset() throws IOException, TranslateException {
        CookingStackExchange trainingSet =
                CookingStackExchange.builder().optUsage(Dataset.Usage.TEST).build();
        trainingSet.prepare();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }

    @Test(expectedExceptions = IOException.class)
    public void testValidateDataset() throws IOException, TranslateException {
        CookingStackExchange trainingSet =
                CookingStackExchange.builder().optUsage(Dataset.Usage.VALIDATION).build();
        trainingSet.prepare();
    }
}
