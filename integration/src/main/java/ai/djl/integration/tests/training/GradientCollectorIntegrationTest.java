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
import ai.djl.integration.util.Assertions;
import ai.djl.mxnet.engine.MxGradientCollector;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import org.testng.Assert;
import org.testng.annotations.Test;

public class GradientCollectorIntegrationTest {

    @Test
    public void testAutograd() {
        try (NDManager manager = NDManager.newBaseManager();
                MxGradientCollector gradCol = new MxGradientCollector()) {
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            NDArray expected = manager.create(new float[] {2, 3, -4, 2, 3, -4}, new Shape(2, 3));
            lhs.attachGradient();
            // autograd automatically set recording and training during initialization
            Assert.assertTrue(MxGradientCollector.isRecording());
            Assert.assertTrue(MxGradientCollector.isTraining());

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

    @Test
    public void testTrain() {
        int numOfData = 1000;
        int batchSize = 10;
        int epochs = 10;

        Optimizer optimizer =
                new Sgd.Builder()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(.03f))
                        .build();

        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss()).setOptimizer(optimizer);

        try (Model model = Model.newInstance()) {
            Linear block = new Linear.Builder().setOutChannels(1).build();
            model.setBlock(block);

            NDManager manager = model.getNDManager();

            NDArray weight = manager.create(new float[] {2.f, -3.4f}, new Shape(2, 1));
            float bias = 4.2f;
            NDArray data = manager.randomNormal(new Shape(numOfData, weight.size(0)));
            // y = w * x + b
            NDArray label = data.dot(weight).add(bias);
            // add noise
            label.add(
                    manager.randomNormal(
                            0, 0.01, label.getShape(), DataType.FLOAT32, manager.getDevice()));

            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .optLabels(label)
                            .setSampling(batchSize, false)
                            .build();
            float lossValue;
            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(batchSize, weight.size(0));
                trainer.initialize(inputShape);

                for (int epoch = 0; epoch < epochs; epoch++) {
                    trainer.resetTrainingMetrics();
                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        trainer.trainBatch(batch);
                        trainer.step();
                        batch.close();
                    }
                }
                lossValue = trainer.getLoss().getValue();
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
