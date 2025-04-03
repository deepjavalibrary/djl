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
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.BulkDataIterable;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.StackBatchifier;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

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

    @SuppressWarnings("unchecked")
    @Test
    public void testRandomSamplingCoversAllIndices() throws IOException, TranslateException {
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

            List<Long> randomizedIndices = new ArrayList<>();
            try (Trainer trainer = model.newTrainer(config)) {
                for (Batch batch : trainer.iterateDataset(mnist)) {
                    randomizedIndices.addAll((List<Long>) batch.getIndices());
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                    if (randomizedIndices.size() == 10000) {
                        // 312 * 32 + 16 = 10000, so last batch has only 16 items left
                        Assert.assertEquals(
                                new Shape(16, 1, 28, 28), batch.getData().get(0).getShape());
                    } else {
                        Assert.assertEquals(
                                new Shape(32, 1, 28, 28), batch.getData().get(0).getShape());
                    }
                    batch.close();
                }
            }
            Assert.assertEquals(10000, randomizedIndices.size());
            Assert.assertFalse(BulkDataIterable.isRange(randomizedIndices));
            Collections.sort(randomizedIndices);
            Assert.assertTrue(BulkDataIterable.isRange(randomizedIndices));
        }
    }

    @Test
    public void testBulkEqualsNonBulk() throws IOException, TranslateException {
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");

            // use BulkDataIterable
            Mnist mnistBulk =
                    Mnist.builder()
                            .optManager(manager)
                            .optUsage(Dataset.Usage.TEST)
                            .optRepository(repository)
                            .setSampling(32, false)
                            .build();

            // use only DataIterable (as not using Batchifier.STACK)
            Mnist mnist =
                    Mnist.builder()
                            .optManager(manager)
                            .optUsage(Dataset.Usage.TEST)
                            .optRepository(repository)
                            .setSampling(32, false)
                            .optLabelBatchifier(new StackBatchifier())
                            .build();

            try (Trainer trainer = model.newTrainer(config)) {
                Iterator<Batch> iterable = trainer.iterateDataset(mnist).iterator();
                Iterator<Batch> bulkIterable = trainer.iterateDataset(mnistBulk).iterator();
                Assert.assertFalse(iterable instanceof BulkDataIterable);
                Assert.assertTrue(bulkIterable instanceof BulkDataIterable);
                while (iterable.hasNext()) {
                    Batch batch = iterable.next();
                    Assert.assertTrue(bulkIterable.hasNext());
                    Batch batchBulk = bulkIterable.next();
                    Assert.assertEquals(batch.getIndices(), batchBulk.getIndices());
                    Assert.assertEquals(batch.getSize(), batchBulk.getSize());
                    Assert.assertEquals(batch.getData(), batchBulk.getData());
                    Assert.assertEquals(batch.getLabels(), batchBulk.getLabels());
                    batch.close();
                    batchBulk.close();
                }
                Assert.assertFalse(bulkIterable.hasNext());
            }
        }
    }
}
