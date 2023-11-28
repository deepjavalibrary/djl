/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.integration.tests.training.listener;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.integration.util.TestUtils;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.EarlyStoppingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.io.IOException;
import java.time.Duration;

public class EarlyStoppingListenerTest {

    private final Optimizer sgd =
            Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.1f)).build();

    private NDManager manager;
    private Mnist testMnistDataset;
    private Mnist trainMnistDataset;

    @BeforeTest
    public void setUp() throws IOException, TranslateException {
        manager = NDManager.newBaseManager(TestUtils.getEngine());
        testMnistDataset =
                Mnist.builder()
                        .optUsage(Dataset.Usage.TEST)
                        .optManager(manager)
                        .optLimit(8)
                        .setSampling(8, false)
                        .build();
        testMnistDataset.prepare();

        trainMnistDataset =
                Mnist.builder()
                        .optUsage(Dataset.Usage.TRAIN)
                        .optManager(manager)
                        .optLimit(16)
                        .setSampling(8, false)
                        .build();
        trainMnistDataset.prepare();
    }

    @AfterTest
    public void closeResources() {
        manager.close();
    }

    @Test
    public void testEarlyStoppingStopsOnEpoch2() throws Exception {
        Mlp mlpModel = new Mlp(784, 1, new int[] {256}, Activation::relu);

        try (Model model = Model.newInstance("lin-reg", TestUtils.getEngine())) {
            model.setBlock(mlpModel);

            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss())
                            .optOptimizer(sgd)
                            .addTrainingListeners(TrainingListener.Defaults.logging())
                            .addTrainingListeners(
                                    EarlyStoppingListener.builder()
                                            .optEpochPatience(1)
                                            .optEarlyStopPctImprovement(99)
                                            .optMaxDuration(Duration.ofMinutes(1))
                                            .optMinEpochs(1)
                                            .build());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 784));
                Metrics metrics = new Metrics();
                trainer.setMetrics(metrics);

                try {
                    // Set epoch to 5 as we expect the early stopping to stop after the second epoch
                    EasyTrain.fit(trainer, 5, trainMnistDataset, testMnistDataset);
                } catch (EarlyStoppingListener.EarlyStoppedException e) {
                    Assert.assertEquals(
                            e.getMessage(), "failed to achieve 99.0% improvement 1 times in a row");
                    Assert.assertEquals(e.getStopEpoch(), 2);
                }

                TrainingResult trainingResult = trainer.getTrainingResult();
                Assert.assertEquals(trainingResult.getEpoch(), 2);
            }
        }
    }

    @Test
    public void testEarlyStoppingStopsOnEpoch3AsMinEpochsIs3() throws Exception {
        Mlp mlpModel = new Mlp(784, 1, new int[] {256}, Activation::relu);

        try (Model model = Model.newInstance("lin-reg", TestUtils.getEngine())) {
            model.setBlock(mlpModel);

            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss())
                            .optOptimizer(sgd)
                            .addTrainingListeners(TrainingListener.Defaults.logging())
                            .addTrainingListeners(
                                    EarlyStoppingListener.builder()
                                            .optEpochPatience(1)
                                            .optEarlyStopPctImprovement(50)
                                            .optMaxMillis(60_000)
                                            .optMinEpochs(3)
                                            .build());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 784));
                Metrics metrics = new Metrics();
                trainer.setMetrics(metrics);

                try {
                    // Set epoch to 5 as we expect the early stopping to stop after the second epoch
                    EasyTrain.fit(trainer, 5, trainMnistDataset, testMnistDataset);
                } catch (EarlyStoppingListener.EarlyStoppedException e) {
                    Assert.assertEquals(
                            e.getMessage(), "failed to achieve 50.0% improvement 1 times in a row");
                    Assert.assertEquals(e.getStopEpoch(), 3);
                }

                TrainingResult trainingResult = trainer.getTrainingResult();
                Assert.assertEquals(trainingResult.getEpoch(), 3);
            }
        }
    }

    @Test
    public void testEarlyStoppingStopsOnEpoch1AsMaxDurationIs1ms() throws Exception {
        Mlp mlpModel = new Mlp(784, 1, new int[] {256}, Activation::relu);

        try (Model model = Model.newInstance("lin-reg", TestUtils.getEngine())) {
            model.setBlock(mlpModel);

            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss())
                            .optOptimizer(sgd)
                            .addTrainingListeners(TrainingListener.Defaults.logging())
                            .addTrainingListeners(
                                    EarlyStoppingListener.builder().optMaxMillis(1).build());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 784));
                Metrics metrics = new Metrics();
                trainer.setMetrics(metrics);

                try {
                    // Set epoch to 10 as we expect the early stopping to stop after the second
                    // epoch
                    EasyTrain.fit(trainer, 10, trainMnistDataset, testMnistDataset);
                } catch (EarlyStoppingListener.EarlyStoppedException e) {
                    Assert.assertTrue(e.getMessage().contains("ms elapsed >= 1 maxMillis"));
                    Assert.assertTrue(e.getStopEpoch() < 10); // Stop epoch is before 10
                }

                TrainingResult trainingResult = trainer.getTrainingResult();
                Assert.assertTrue(trainingResult.getEpoch() < 10); // Stop epoch is before 10
            }
        }
    }
}
