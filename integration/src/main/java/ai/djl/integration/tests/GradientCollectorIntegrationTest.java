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
package ai.djl.integration.tests;

import ai.djl.Model;
import ai.djl.integration.util.Assertions;
import ai.djl.mxnet.engine.MxGradientCollector;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.LossMetric;
import ai.djl.training.optimizer.Nag;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.util.PairList;
import ai.djl.zoo.cv.classification.ResNetV1;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class GradientCollectorIntegrationTest {

    @Test
    public void testAutograd() {
        try (NDManager manager = NDManager.newBaseManager();
                MxGradientCollector gradCol = new MxGradientCollector()) {
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            lhs.attachGradient();
            // autograd automatically set recording and training during initialization
            Assert.assertTrue(MxGradientCollector.isRecording());
            Assert.assertTrue(MxGradientCollector.isTraining());

            NDArray result = NDArrays.dot(lhs, rhs);
            gradCol.backward(result);
        }
    }

    @Test
    public void testTrain() throws IOException {
        int numOfData = 1000;
        int batchSize = 10;
        int epochs = 10;

        Optimizer optimizer =
                new Sgd.Builder()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(.03f))
                        .build();

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, optimizer);

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

            NDArray loss;
            LossMetric lossMetric = new LossMetric("l2loss");

            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .optLabels(label)
                            .setRandomSampling(batchSize)
                            .build();
            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(batchSize, weight.size(0));
                trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

                for (int epoch = 0; epoch < epochs; epoch++) {
                    lossMetric.reset();
                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        try (GradientCollector gradCol = trainer.newGradientCollector()) {

                            NDArray x = batch.getData().head();
                            NDArray y = batch.getLabels().head();
                            NDArray yHat = block.forward(new NDList(x)).head();
                            loss = Loss.l2Loss().getLoss(y, yHat);
                            gradCol.backward(loss);
                        }
                        trainer.step();
                        lossMetric.update(loss);
                        batch.close();
                    }
                }
            }
            float lossValue = lossMetric.getMetric().getValue();
            float expectedLoss = 0.001f;
            Assert.assertTrue(
                    lossValue < expectedLoss,
                    String.format(
                            "Loss did not improve, loss value: %f, expected "
                                    + "max loss value: %f",
                            lossValue, expectedLoss));
        }
    }

    @Test
    public void testTrainResNet() {
        Optimizer optimizer =
                new Nag.Builder()
                        .setRescaleGrad(1.0f / 100)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .setMomentum(0.9f)
                        .build();

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, optimizer);

        Block resNet50 =
                new ResNetV1.Builder()
                        .setImageShape(new Shape(1, 28, 28))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();

        try (Model model = Model.newInstance()) {
            model.setBlock(resNet50);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(100, 1, 28, 28);
                trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

                NDManager manager = trainer.getManager();

                NDArray input = manager.ones(inputShape);
                NDArray label = manager.ones(new Shape(100, 1));
                PairList<String, Parameter> parameters = resNet50.getParameters();
                try (GradientCollector gradCol = trainer.newGradientCollector()) {
                    NDArray pred = trainer.forward(new NDList(input)).head();
                    NDArray loss = Loss.softmaxCrossEntropyLoss().getLoss(label, pred);
                    gradCol.backward(loss);
                }
                trainer.step();
                NDArray expectedAtIndex0 = manager.ones(new Shape(16, 1, 3, 3));
                NDArray expectedAtIndex1 = manager.ones(new Shape(16)).muli(.8577);
                NDArray expectedAtIndex87 = manager.ones(new Shape(32, 32, 3, 3));
                Assert.assertEquals(parameters.get(0).getValue().getArray(), expectedAtIndex0);
                Assertions.assertAlmostEquals(
                        parameters.get(1).getValue().getArray(), expectedAtIndex1);
                Assert.assertEquals(expectedAtIndex87, parameters.get(87).getValue().getArray());
            }
        }
    }
}
