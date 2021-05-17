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
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class EvaluateDatasetTest {

    @Test
    public void testDatasetEvaluation() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Mnist testMnistDataset =
                    Mnist.builder()
                            .optManager(manager)
                            .optUsage(Dataset.Usage.TEST)
                            .setSampling(32, true)
                            .build();

            testMnistDataset.prepare();

            Mlp mlpModel = new Mlp(784, 1, new int[] {256}, Activation::relu);

            Model model = Model.newInstance("lin-reg");
            model.setBlock(mlpModel);

            Loss l2loss = Loss.l2Loss();

            Tracker lrt = Tracker.fixed(0.5f);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(l2loss)
                            .optOptimizer(sgd) // Optimizer (loss function)
                            .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 784));

                Metrics metrics = new Metrics();
                trainer.setMetrics(metrics);

                EasyTrain.evaluateDataset(trainer, testMnistDataset);

                Assert.assertTrue(
                        l2loss.getAccumulator(EvaluatorTrainingListener.VALIDATE_EPOCH) < 25.0f,
                        "dataset L2 loss is more than expected.");
            }

            model.close();
        }
    }
}
