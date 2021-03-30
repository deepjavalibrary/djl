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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class GradientCollectorIntegrationTest {

    @Test
    public void testAutograd() {
        try (Model model = Model.newInstance("model");
                NDManager manager = model.getNDManager()) {
            model.setBlock(Blocks.identityBlock());
            try (Trainer trainer =
                    model.newTrainer(
                            new DefaultTrainingConfig(Loss.l2Loss())
                                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT))) {
                try (GradientCollector gradCol = trainer.newGradientCollector()) {
                    NDArray lhs =
                            manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
                    NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
                    NDArray expected =
                            manager.create(new float[] {2, 3, -4, 2, 3, -4}, new Shape(2, 3));
                    lhs.setRequiresGradient(true);
                    // autograd automatically set recording and training during initialization

                    NDArray result = NDArrays.dot(lhs, rhs);
                    gradCol.backward(result);
                    NDArray grad = lhs.getGradient();
                    Assertions.assertAlmostEquals(grad, expected);
                    // test close and get again
                    grad.close();
                    NDArray grad2 = lhs.getGradient();
                    Assertions.assertAlmostEquals(grad2, expected);
                }
            }
        }
    }

    @Test
    public void testTrain() throws IOException, TranslateException {
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }

        int numOfData = 1000;
        int batchSize = 10;
        int epochs = 10;

        Optimizer optimizer = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(.03f)).build();

        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .addTrainingListeners(new EvaluatorTrainingListener())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer);

        try (Model model = Model.newInstance("linear")) {
            Linear block = Linear.builder().setUnits(1).build();
            model.setBlock(block);

            NDManager manager = model.getNDManager();

            NDArray weight = manager.create(new float[] {2.f, -3.4f}, new Shape(2, 1));
            float bias = 4.2f;
            NDArray data = manager.randomNormal(new Shape(numOfData, weight.size(0)));
            // y = w * x + b
            NDArray label = data.dot(weight).add(bias);
            // add noise
            label.addi(
                    manager.randomNormal(
                            0f, 0.01f, label.getShape(), DataType.FLOAT32, manager.getDevice()));

            int sampling = config.getDevices().length * batchSize;
            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .optLabels(label)
                            .setSampling(sampling, false)
                            .build();
            float lossValue;
            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(sampling, weight.size(0));
                trainer.initialize(inputShape);

                for (int epoch = 0; epoch < epochs; epoch++) {
                    trainer.notifyListeners(listener -> listener.onEpoch(trainer));
                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        EasyTrain.trainBatch(trainer, batch);
                        trainer.step();
                        batch.close();
                    }
                }
                lossValue = trainer.getLoss().getAccumulator(EvaluatorTrainingListener.TRAIN_EPOCH);
            }
            float expectedLoss = 0.001f;
            Assert.assertTrue(
                    lossValue < expectedLoss,
                    String.format(
                            "Loss did not improve, loss value: %f, expected "
                                    + "max loss value: %f",
                            lossValue, expectedLoss));
        }
    }
}
