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
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MnistTest {

    @Test
    public void testMnistLocal() throws IOException, TranslateException {
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
            Mnist mnist =
                    Mnist.builder()
                            .optManager(manager)
                            .optUsage(Dataset.Usage.TEST)
                            .optRepository(repository)
                            .setSampling(32, true)
                            .build();

            try (Trainer trainer = model.newTrainer(config)) {
                Batch batch = trainer.iterateDataset(mnist).iterator().next();
                Assert.assertEquals(batch.getData().size(), 1);
                Assert.assertEquals(batch.getLabels().size(), 1);
                batch.close();
            }
        }
    }

    @Test
    public void testMnistRemote() throws IOException, TranslateException {
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            Mnist mnist =
                    Mnist.builder()
                            .optManager(manager)
                            .optUsage(Dataset.Usage.TEST)
                            .setSampling(32, true)
                            .build();

            try (Trainer trainer = model.newTrainer(config)) {
                Batch batch = trainer.iterateDataset(mnist).iterator().next();
                Assert.assertEquals(batch.getData().size(), 1);
                Assert.assertEquals(batch.getLabels().size(), 1);
                batch.close();
            }
        }
    }
}
