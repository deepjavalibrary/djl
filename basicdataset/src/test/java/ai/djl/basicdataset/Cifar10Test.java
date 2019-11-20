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
package ai.djl.basicdataset;

import ai.djl.Model;
import ai.djl.nn.Blocks;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset.Usage;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class Cifar10Test {

    @Test
    public void testCifar10Local() throws IOException {
        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.softmaxCrossEntropyLoss());

        try (Model model = Model.newInstance()) {
            model.setBlock(Blocks.identityBlock());

            Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
            Cifar10 cifar10 =
                    new Cifar10.Builder()
                            .setManager(model.getNDManager())
                            .optUsage(Usage.TEST)
                            .optRepository(repository)
                            .setSampling(32, true)
                            .build();

            cifar10.prepare();
            try (Trainer trainer = model.newTrainer(config)) {
                for (Batch batch : trainer.iterateDataset(cifar10)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                    batch.close();
                }
            }
        }
    }

    @Test
    public void testCifar10Remote() throws IOException {
        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.softmaxCrossEntropyLoss());

        try (Model model = Model.newInstance()) {
            model.setBlock(Blocks.identityBlock());

            Cifar10 cifar10 =
                    new Cifar10.Builder()
                            .setManager(model.getNDManager())
                            .optUsage(Usage.TEST)
                            .setSampling(32, true)
                            .build();

            cifar10.prepare();
            try (Trainer trainer = model.newTrainer(config)) {
                for (Batch batch : trainer.iterateDataset(cifar10)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                    batch.close();
                }
            }
        }
    }
}
