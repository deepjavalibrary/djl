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
import org.apache.mxnet.engine.MxGradient;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.Block;
import software.amazon.ai.Parameter;
import software.amazon.ai.SequentialBlock;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.MnistUtils;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.Gradient;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.dataset.ArrayDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.initializer.NormalInitializer;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.lrscheduler.LrScheduler;
import software.amazon.ai.util.PairList;
import software.amazon.ai.zoo.cv.image_classification.ResNetV1;

public class GradientCollectorIntegrationTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", GradientCollectorIntegrationTest.class.getName()};
        JnaUtils.setNumpyMode(true);
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
        JnaUtils.setNumpyMode(false);
    }

    @RunAsTest
    public void testAutograd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager();
                Gradient.Collector gradCol = Gradient.newCollector()) {
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            gradCol.collectFor(lhs);
            // autograd automatically set recording and training during initialization
            if (gradCol instanceof MxGradient.Collector) {
                Assertions.assertTrue(MxGradient.isRecording());
                Assertions.assertTrue(MxGradient.isTraining());
            }
            NDArray result = NDArrays.mmul(lhs, rhs);
            gradCol.collect(result);
        }
    }

    @RunAsTest
    public void testTrain() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
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
                            0, 0.01, label.getShape(), DataType.FLOAT32, manager.getContext()));
            Linear block = new Linear.Builder().setOutChannels(1).build();
            block.setInitializer(manager, Initializer.ONES);

            Optimizer optimizer =
                    new Sgd.Builder(block.getParameters())
                            .setRescaleGrad(1.0f / batchSize)
                            .setLrScheduler(LrScheduler.fixedLR(.03f))
                            .build();
            NDArray loss;
            LossMetric lossMetric = new LossMetric("l2loss");

            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .setLabels(label)
                            .setDataLoadingProperty(true, batchSize, false)
                            .build();

            for (int epoch = 0; epoch < epochs; epoch++) {
                lossMetric.reset();
                for (Record record : dataset.getRecords()) {
                    try (Gradient.Collector gradCol = Gradient.newCollector()) {
                        Gradient.OptimizerKey optKey = gradCol.collectFor(optimizer);

                        NDArray x = record.getData().head();
                        NDArray y = record.getLabels().head();
                        NDArray yHat = block.forward(x);
                        loss = Loss.l2Loss(y, yHat, 1, 0);
                        optimizer.step(gradCol.collect(loss).get(optKey));
                    }
                    lossMetric.update(loss);
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
    public void testTrainMnist() throws IOException, FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            SequentialBlock mlp = new SequentialBlock();
            mlp.add(new Linear.Builder().setOutChannels(128).build());
            mlp.add(Activation.reluBlock());
            mlp.add(new Linear.Builder().setOutChannels(64).build());
            mlp.add(Activation.reluBlock());
            mlp.add(new Linear.Builder().setOutChannels(10).build());
            mlp.setInitializer(manager, new NormalInitializer(0.01));

            int numEpoch = 3;
            MnistUtils.trainMnist(mlp, manager, numEpoch, 0.3f, 0.9f);
        }
    }

    @RunAsTest
    public void testTrainResNet() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block resNet50 =
                    new ResNetV1.Builder()
                            .setImageShape(new Shape(1, 28, 28))
                            .setNumLayers(50)
                            .setOutSize(10)
                            .build();
            resNet50.setInitializer(manager, Initializer.ONES);
            Optimizer optimizer =
                    new Sgd.Builder(resNet50.getParameters())
                            .setRescaleGrad(1.0f / 100)
                            .setLrScheduler(LrScheduler.fixedLR(0.1f))
                            .setMomentum(0.9f)
                            .build();
            NDArray input = manager.ones(new Shape(100, 1, 28, 28));
            NDArray label = manager.ones(new Shape(100, 1));
            try (Gradient.Collector gradCol = Gradient.newCollector()) {
                Gradient.OptimizerKey optKey = gradCol.collectFor(optimizer);
                NDArray pred = resNet50.forward(new NDList(input)).head();
                NDArray loss = Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
                optimizer.step(gradCol.collect(loss).get(optKey));
            }
            PairList<String, Parameter> parameters = optimizer.getParameters();
            NDArray expectedAtIndex0 = manager.ones(new Shape(16, 1, 3, 3));
            NDArray expectedAtIndex1 = manager.ones(new Shape(16)).muli(1.7576532f);
            NDArray expectedAtIndex87 = manager.ones(new Shape(32, 32, 3, 3));
            Assertions.assertEquals(expectedAtIndex0, parameters.get(0).getValue().getArray());
            Assertions.assertEquals(expectedAtIndex1, parameters.get(1).getValue().getArray());
            Assertions.assertEquals(expectedAtIndex87, parameters.get(87).getValue().getArray());
        }
    }
}
