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
package ai.djl.mxnet.dataset;

import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;
import software.amazon.ai.Model;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.Dataset.Usage;
import software.amazon.ai.training.initializer.Initializer;

public class Cifar10Test {

    @Test
    public void testCifar10Local() throws IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES);

        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.IDENTITY_BLOCK);

            Repository repository = Repository.newInstance("test", "src/test/resources/repo");
            Cifar10 cifar10 =
                    new Cifar10.Builder()
                            .setManager(model.getNDManager())
                            .setUsage(Usage.TEST)
                            .optRepository(repository)
                            .setRandomSampling(32)
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
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES);

        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.IDENTITY_BLOCK);

            Cifar10 cifar10 =
                    new Cifar10.Builder()
                            .setManager(model.getNDManager())
                            .setUsage(Usage.TEST)
                            .setRandomSampling(32)
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
