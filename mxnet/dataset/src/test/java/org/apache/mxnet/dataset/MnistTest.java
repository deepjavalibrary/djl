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
import software.amazon.ai.Model;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.BlockFactory;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.dataset.ArrayDataset;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.Dataset;

public class MnistTest {

    @Test
    public void testMnistLocal() throws IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            model.setBlock(factory.createIdentityBlock());

            NDManager manager = model.getNDManager();
            Repository repository = Repository.newInstance("test", "src/test/resources/repo");
            Mnist mnist =
                    new Mnist.Builder()
                            .setManager(manager)
                            .setUsage(Dataset.Usage.TEST)
                            .optRepository(repository)
                            .setSampling(32)
                            .build();

            mnist.prepare();
            try (Trainer<NDList, NDList, NDList> trainer =
                    model.newTrainer(new ArrayDataset.DefaultTranslator())) {
                for (Batch batch : trainer.iterateDataset(mnist)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                    batch.close();
                }
            }
        }
    }

    @Test
    public void testMnistRemote() throws IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            model.setBlock(factory.createIdentityBlock());

            NDManager manager = model.getNDManager();
            Mnist mnist =
                    new Mnist.Builder()
                            .setManager(manager)
                            .setUsage(Dataset.Usage.TEST)
                            .setSampling(32)
                            .build();

            mnist.prepare();
            try (Trainer<NDList, NDList, NDList> trainer =
                    model.newTrainer(new ArrayDataset.DefaultTranslator())) {
                for (Batch batch : trainer.iterateDataset(mnist)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                    batch.close();
                }
            }
        }
    }
}
