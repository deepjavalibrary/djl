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
package software.amazon.ai.integration.tests;

import java.io.IOException;
import org.apache.mxnet.engine.MxGradientCollector;
import org.testng.annotations.Test;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.TrainingController;
import software.amazon.ai.training.dataset.ArrayDataset;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.training.optimizer.Nag;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.learningrate.LearningRateTracker;
import software.amazon.ai.util.PairList;
import software.amazon.ai.zoo.cv.classification.ResNetV1;

public class GradientCollectorIntegrationTest {

    @Test
    public void testAutograd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager();
                MxGradientCollector gradCol = new MxGradientCollector()) {
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            lhs.attachGradient();
            // autograd automatically set recording and training during initialization
            Assertions.assertTrue(MxGradientCollector.isRecording());
            Assertions.assertTrue(MxGradientCollector.isTraining());

            NDArray result = NDArrays.mmul(lhs, rhs);
            gradCol.backward(result);
        }
    }

    @Test
    public void testTrain() throws FailedTestException, IOException {
        int numOfData = 1000;
        int batchSize = 10;
        int epochs = 10;

        Optimizer optimizer =
                new Sgd.Builder()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLR(.03f))
                        .build();

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false, optimizer);

        try (Model model = Model.newInstance()) {
            Linear block = new Linear.Builder().setOutChannels(1).build();
            model.setBlock(block);

            NDManager manager = model.getNDManager();

            NDArray weight = manager.create(new float[] {2.f, -3.4f}, new Shape(2, 1));
            float bias = 4.2f;
            NDArray data = manager.randomNormal(new Shape(numOfData, weight.size(0)));
            // y = w * x + b
            NDArray label = data.mmul(weight).add(bias);
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
                            .setSampling(batchSize, true, true)
                            .build();
            try (Trainer trainer = model.newTrainer(config)) {
                for (int epoch = 0; epoch < epochs; epoch++) {
                    lossMetric.reset();
                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        try (GradientCollector gradCol = trainer.newGradientCollector()) {

                            NDArray x = batch.getData().head();
                            NDArray y = batch.getLabels().head();
                            NDArray yHat = block.forward(new NDList(x)).head();
                            loss = Loss.l2Loss(y, yHat, 1, 0);
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
            Assertions.assertTrue(
                    lossValue < expectedLoss,
                    String.format(
                            "Loss did not improve, loss value: %f, expected "
                                    + "max loss value: %f",
                            lossValue, expectedLoss));
        }
    }

    @Test
    public void testTrainResNet() throws FailedTestException {
        Optimizer optimizer =
                new Nag.Builder()
                        .setRescaleGrad(1.0f / 100)
                        .setLearningRateTracker(LearningRateTracker.fixedLR(0.1f))
                        .setMomentum(0.9f)
                        .build();

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false, optimizer);

        Block resNet50 =
                new ResNetV1.Builder()
                        .setImageShape(new Shape(1, 28, 28))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();

        try (Model model = Model.newInstance()) {
            model.setBlock(resNet50);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                NDArray input = manager.ones(new Shape(100, 1, 28, 28));
                NDArray label = manager.ones(new Shape(100, 1));
                PairList<String, Parameter> parameters = resNet50.getParameters();
                TrainingController controller =
                        new TrainingController(
                                parameters, optimizer, new Device[] {manager.getDevice()});
                try (GradientCollector gradCol = trainer.newGradientCollector()) {
                    NDArray pred = trainer.forward(new NDList(input)).head();
                    NDArray loss =
                            Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
                    gradCol.backward(loss);
                }
                controller.step();
                NDArray expectedAtIndex0 = manager.ones(new Shape(16, 1, 3, 3));
                NDArray expectedAtIndex1 = manager.ones(new Shape(16)).muli(1.7576532f);
                NDArray expectedAtIndex87 = manager.ones(new Shape(32, 32, 3, 3));
                Assertions.assertEquals(expectedAtIndex0, parameters.get(0).getValue().getArray());
                Assertions.assertEquals(expectedAtIndex1, parameters.get(1).getValue().getArray());
                Assertions.assertEquals(
                        expectedAtIndex87, parameters.get(87).getValue().getArray());
                controller.close();
            }
        }
    }
}
