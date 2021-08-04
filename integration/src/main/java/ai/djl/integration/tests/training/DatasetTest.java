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
package ai.djl.integration.tests.training;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.BatchSampler;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.RandomSampler;
import ai.djl.training.dataset.SequenceSampler;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DatasetTest {

    @Test
    public void testSequenceSampler() throws IOException, TranslateException {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();

            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(
                                    manager.arange(
                                            0, 100, 1, DataType.INT64, manager.defaultDevice()))
                            .setSampling(new BatchSampler(new SequenceSampler(), 1, false))
                            .build();

            List<Long> original = new ArrayList<>();
            try (Trainer trainer = model.newTrainer(config())) {
                trainer.iterateDataset(dataset)
                        .iterator()
                        .forEachRemaining(
                                record ->
                                        original.add(
                                                record.getData().singletonOrThrow().getLong()));
                List<Long> expected = LongStream.range(0, 100).boxed().collect(Collectors.toList());
                Assert.assertEquals(original, expected, "SequentialSampler test failed");
            }
        }
    }

    @Test
    public void testRandomSampler() throws IOException, TranslateException {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();

            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(
                                    manager.arange(
                                            0, 10, 1, DataType.INT64, manager.defaultDevice()))
                            .setSampling(new BatchSampler(new RandomSampler(), 1, false))
                            .build();
            List<Long> original = new ArrayList<>();
            try (Trainer trainer = model.newTrainer(config())) {
                trainer.iterateDataset(dataset)
                        .iterator()
                        .forEachRemaining(
                                record ->
                                        original.add(
                                                record.getData().singletonOrThrow().getLong()));
                Assert.assertEquals(original.size(), 10, "RandomSampler test failed");
            }
        }
    }

    @Test
    public void testBatchSampler() throws IOException, TranslateException {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();

            NDArray data = manager.arange(0, 100, 1, DataType.INT64, manager.defaultDevice());

            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .setSampling(new BatchSampler(new SequenceSampler(), 27, false))
                            .build();
            List<long[]> originalList = new ArrayList<>();
            try (Trainer trainer = model.newTrainer(config())) {
                trainer.iterateDataset(dataset)
                        .iterator()
                        .forEachRemaining(
                                record ->
                                        originalList.add(
                                                record.getData().singletonOrThrow().toLongArray()));
                Assert.assertEquals(originalList.size(), 4, "size of BatchSampler is not correct");
                long[] expected = LongStream.range(0, 27).toArray();
                Assert.assertTrue(
                        Arrays.equals(originalList.get(0), expected),
                        "data from BatchSampler is not correct");
            }

            ArrayDataset dataset2 =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .setSampling(new BatchSampler(new RandomSampler(), 33, true))
                            .build();
            List<long[]> originalList2 = new ArrayList<>();
            try (Trainer trainer = model.newTrainer(config())) {
                trainer.iterateDataset(dataset2)
                        .iterator()
                        .forEachRemaining(
                                record ->
                                        originalList2.add(
                                                record.getData().singletonOrThrow().toLongArray()));
                Assert.assertEquals(originalList2.size(), 3, "size of BatchSampler is not correct");
            }

            // test case when dataset is smaller than batchSize, dropLast=true
            ArrayDataset dataset3 =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .setSampling(new BatchSampler(new SequenceSampler(), 101, true))
                            .build();
            List<long[]> originalList3 = new ArrayList<>();
            try (Trainer trainer = model.newTrainer(config())) {
                trainer.iterateDataset(dataset3)
                        .iterator()
                        .forEachRemaining(
                                record ->
                                        originalList3.add(
                                                record.getData().singletonOrThrow().toLongArray()));
                Assert.assertTrue(originalList3.isEmpty(), "size of BatchSampler is not correct");
            }

            // test case when dataset is smaller than batchSize, dropLast=false
            ArrayDataset dataset4 =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .setSampling(new BatchSampler(new SequenceSampler(), 101, false))
                            .build();
            List<long[]> originalList4 = new ArrayList<>();
            try (Trainer trainer = model.newTrainer(config())) {
                trainer.iterateDataset(dataset4)
                        .iterator()
                        .forEachRemaining(
                                record ->
                                        originalList4.add(
                                                record.getData().singletonOrThrow().toLongArray()));
                Assert.assertTrue(
                        originalList4.size() == 1 && originalList4.get(0).length == 100,
                        "size of BatchSampler is not correct");
            }
        }
    }

    @Test
    public void testArrayDataset() throws IOException, TranslateException {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            // single case: one data, one label
            NDArray data = manager.arange(200).reshape(100, 2);
            NDArray label = manager.arange(100).reshape(100);
            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .optLabels(label)
                            .setSampling(20, false)
                            .build();

            int index = 0;
            try (Trainer trainer = model.newTrainer(config())) {
                for (Batch batch : trainer.iterateDataset(dataset)) {
                    Assert.assertEquals(
                            batch.getData().singletonOrThrow(),
                            manager.arange(2 * index, 2 * index + 40).reshape(20, 2));
                    Assert.assertEquals(
                            batch.getLabels().singletonOrThrow(),
                            manager.arange(index, index + 20).reshape(20));
                    index += 20;
                }

                dataset =
                        new ArrayDataset.Builder()
                                .setData(data)
                                .optLabels(label)
                                .setSampling(15, false)
                                .build();
                index = 0;
                for (Batch batch : trainer.iterateDataset(dataset)) {
                    if (index != 90) {
                        Assert.assertEquals(
                                batch.getData().singletonOrThrow(),
                                manager.arange(2 * index, 2 * index + 30).reshape(15, 2));
                        Assert.assertEquals(
                                batch.getLabels().singletonOrThrow(),
                                manager.arange(index, index + 15).reshape(15));
                    } else {
                        // last batch
                        Assert.assertEquals(
                                batch.getData().singletonOrThrow(),
                                manager.arange(2 * index, 2 * index + 20).reshape(10, 2));
                        Assert.assertEquals(
                                batch.getLabels().singletonOrThrow(),
                                manager.arange(index, index + 10).reshape(10));
                    }
                    index += 15;
                }
                // multiple data, 0 labels test case
                NDArray data2 = manager.arange(300).reshape(100, 3);
                index = 0;
                dataset =
                        new ArrayDataset.Builder()
                                .setData(data, data2)
                                .setSampling(10, false)
                                .build();

                for (Batch batch : trainer.iterateDataset(dataset)) {
                    Assert.assertEquals(batch.getData().size(), 2);
                    Assert.assertEquals(
                            batch.getData().head(),
                            manager.arange(2 * index, 2 * index + 20).reshape(10, 2));
                    Assert.assertEquals(
                            batch.getData().get(1),
                            manager.arange(3 * index, 3 * index + 30).reshape(10, 3));
                    Assert.assertEquals(batch.getLabels().size(), 0);
                    index += 10;
                }
            }
        }
    }

    @Test
    public void testMultithreading() throws IOException, InterruptedException, TranslateException {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());
            NDManager manager = model.getNDManager();

            ExecutorService executor = Executors.newFixedThreadPool(5);

            Cifar10 cifar10 =
                    Cifar10.builder()
                            .optManager(manager)
                            .setSampling(2, true)
                            .optUsage(Dataset.Usage.TEST)
                            // you could start trying prefetchNumber with 2 * number of threads.
                            // This number should be adjusted based on your machines and data.
                            .optPrefetchNumber(4)
                            .optLimit(32)
                            .build();

            TrainingConfig threadedConfig = config().optExecutorService(executor);
            try (Trainer trainer = model.newTrainer(threadedConfig)) {
                for (Batch batch : trainer.iterateDataset(cifar10)) {
                    batch.close();
                }
                // user have to shutdown the executor
                executor.shutdown();
                executor.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
            }
        }
    }

    @Test
    public void testDatasetToArray() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            RandomAccessDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(manager.ones(new Shape(5, 4)))
                            .setSampling(32, false)
                            .optLabels(manager.zeros(new Shape(5, 2)))
                            .build();
            Pair<Number[][], Number[][]> converted = dataset.toArray();
            Number[][] data = converted.getKey();
            Number[][] labels = converted.getValue();

            Assert.assertEquals(data.length, 5);
            Assert.assertEquals(labels.length, 5);

            Assert.assertEquals(data[0].length, 4);
            Assert.assertEquals(labels[0].length, 2);

            Assert.assertEquals(data[0][0], 1f);
            Assert.assertEquals(labels[0][0], 0f);
        }
    }

    private DefaultTrainingConfig config() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
    }
}
