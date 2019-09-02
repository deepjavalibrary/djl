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
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.dataset.Dataset.Usage;
import software.amazon.ai.training.dataset.Record;

public class Cifar10Test {

    @Test
    public void testCifar10Local() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Repository repository = Repository.newInstance("test", "src/test/resources/repo");
            Cifar10 cifar10 =
                    new Cifar10.Builder()
                            .setManager(manager)
                            .setUsage(Usage.TEST)
                            .optRepository(repository)
                            .setSampling(32)
                            .build();

            cifar10.prepare();
            try (Trainer<NDArray, NDArray, NDArray> trainer =
                    Trainer.newInstance(
                            Block.IDENTITY_BLOCK, new SimpleDataset.DefaultTranslator())) {
                for (Record batch : trainer.trainDataset(cifar10)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                }
            }
        }
    }

    @Test
    public void testCifar10Remote() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Cifar10 cifar10 =
                    new Cifar10.Builder()
                            .setManager(manager)
                            .setUsage(Usage.TEST)
                            .setSampling(32)
                            .build();

            cifar10.prepare();
            try (Trainer<NDArray, NDArray, NDArray> trainer =
                    Trainer.newInstance(
                            Block.IDENTITY_BLOCK, new SimpleDataset.DefaultTranslator())) {
                for (Record batch : trainer.trainDataset(cifar10)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                }
            }
        }
    }
}
