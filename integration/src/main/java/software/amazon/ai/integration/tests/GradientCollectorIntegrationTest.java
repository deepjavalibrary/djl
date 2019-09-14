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
import java.util.Arrays;
import java.util.stream.Stream;
import org.apache.mxnet.engine.MxGradientCollector;
import software.amazon.ai.Model;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.BlockFactory;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.Trainer;
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

    public static void main(String[] args) {
        String[] cmd = {"-c", GradientCollectorIntegrationTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testAutograd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager();
                GradientCollector gradCol = GradientCollector.newInstance()) {
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            lhs.attachGradient();
            // autograd automatically set recording and training during initialization
            if (gradCol instanceof MxGradientCollector) {
                Assertions.assertTrue(MxGradientCollector.isRecording());
                Assertions.assertTrue(MxGradientCollector.isTraining());
            }
            NDArray result = NDArrays.mmul(lhs, rhs);
            gradCol.backward(result);
        }
    }

    @RunAsTest
    public void testTrain() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            NDManager manager = model.getNDManager();

            int numOfData = 1000;
            int batchSize = 10;
            int epochs = 10;

            NDArray weight = manager.create(new float[] {2.f, -3.4f}, new Shape(2, 1));
            float bias = 4.2f;
            NDArray data = manager.randomNormal(new Shape(numOfData, weight.size(0)));
            // y = w * x + b
            NDArray label = data.mmul(weight).add(bias);
            // add noise
            label.add(
                    manager.randomNormal(
                            0, 0.01, label.getShape(), DataType.FLOAT32, manager.getDevice()));
            Linear block = new Linear.Builder().setFactory(factory).setOutChannels(1).build();
            model.setBlock(block);

            model.setInitializer(Initializer.ONES);

            Optimizer optimizer =
                    new Sgd.Builder()
                            .setRescaleGrad(1.0f / batchSize)
                            .setLearningRateTracker(LearningRateTracker.fixedLR(.03f))
                            .build();
            NDArray loss;
            LossMetric lossMetric = new LossMetric("l2loss");

            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .optLabels(label)
                            .setSampling(batchSize, true, true)
                            .build();
            try (Trainer<NDList, NDList, NDList> trainer =
                    model.newTrainer(new ArrayDataset.DefaultTranslator(), optimizer)) {
                for (int epoch = 0; epoch < epochs; epoch++) {
                    lossMetric.reset();
                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        try (GradientCollector gradCol = GradientCollector.newInstance()) {

                            NDArray x = batch.getData().head();
                            NDArray y = batch.getLabels().head();
                            NDArray yHat = block.forward(x);
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

    @RunAsTest
    public void testTrainResNet() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            NDManager manager = model.getNDManager();

            Block resNet50 =
                    new ResNetV1.Builder()
                            .setFactory(factory)
                            .setImageShape(new Shape(1, 28, 28))
                            .setNumLayers(50)
                            .setOutSize(10)
                            .build();

            model.setBlock(resNet50);
            model.setInitializer(Initializer.ONES);
            Optimizer optimizer =
                    new Nag.Builder()
                            .setRescaleGrad(1.0f / 100)
                            .setLearningRateTracker(LearningRateTracker.fixedLR(0.1f))
                            .setMomentum(0.9f)
                            .build();
            NDArray input = manager.ones(new Shape(100, 1, 28, 28));
            NDArray label = manager.ones(new Shape(100, 1));

            PairList<String, Parameter> parameters = resNet50.getParameters();

            TrainingController controller = new TrainingController(parameters, optimizer);
            try (GradientCollector gradCol = GradientCollector.newInstance()) {
                NDArray pred = resNet50.forward(new NDList(input)).head();
                NDArray loss = Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
                gradCol.backward(loss);
            }
            controller.step();
            NDArray expectedAtIndex0 = manager.ones(new Shape(16, 1, 3, 3));
            NDArray expectedAtIndex1 = manager.ones(new Shape(16)).muli(1.7576532f);
            NDArray expectedAtIndex87 = manager.ones(new Shape(32, 32, 3, 3));
            Assertions.assertEquals(expectedAtIndex0, parameters.get(0).getValue().getArray());
            Assertions.assertEquals(expectedAtIndex1, parameters.get(1).getValue().getArray());
            Assertions.assertEquals(expectedAtIndex87, parameters.get(87).getValue().getArray());
            controller.close();
        }
    }
}
