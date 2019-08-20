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
package org.apache.mxnet.dataset;

import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.dataset.Record;

public class MnistTest {

    @Test
    public void testMnistLocal() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Repository repository = Repository.newInstance("test", "src/test/resources/repo");
            Mnist mnist =
                    new Mnist.Builder(manager, repository)
                            .setUsage(Dataset.Usage.TEST)
                            .setDataLoadingProperty(false, 32, false)
                            .build();
            for (Record record : mnist.getRecords()) {
                Assert.assertEquals(record.getData().size(), 1);
                Assert.assertEquals(record.getLabels().size(), 1);
            }
        }
    }

    @Test
    public void testMnistRemote() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Mnist mnist =
                    new Mnist.Builder(manager)
                            .setUsage(Dataset.Usage.TEST)
                            .setDataLoadingProperty(false, 32, false)
                            .build();
            for (Record record : mnist.getRecords()) {
                Assert.assertEquals(record.getData().size(), 1);
                Assert.assertEquals(record.getLabels().size(), 1);
            }
        }
    }
}
