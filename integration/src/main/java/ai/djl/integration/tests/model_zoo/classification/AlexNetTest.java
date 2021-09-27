/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.integration.tests.model_zoo.classification;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.AlexNet;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Batchifier;
import ai.djl.util.PairList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AlexNetTest {

    @Test
    public void testTrainWithDefaultChannels() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2));

        Block alexNet = AlexNet.builder().build();
        try (Model model = Model.newInstance("alexnet")) {
            model.setBlock(alexNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 1, 224, 224);
                NDManager manager = trainer.getManager();
                trainer.initialize(inputShape);

                NDArray input = manager.ones(inputShape);
                NDArray label = manager.ones(new Shape(batchSize, 1));
                Batch batch =
                        new Batch(
                                manager.newSubManager(),
                                new NDList(input),
                                new NDList(label),
                                batchSize,
                                Batchifier.STACK,
                                Batchifier.STACK,
                                0,
                                0);
                PairList<String, Parameter> parameters = alexNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(),
                        new Shape(96, 1, 11, 11));
                Assert.assertEquals(
                        parameters.get(2).getValue().getArray().getShape(),
                        new Shape(256, 96, 5, 5));
                Assert.assertEquals(
                        parameters.get(4).getValue().getArray().getShape(),
                        new Shape(384, 256, 3, 3));
                Assert.assertEquals(
                        parameters.get(6).getValue().getArray().getShape(),
                        new Shape(384, 384, 3, 3));
                Assert.assertEquals(
                        parameters.get(8).getValue().getArray().getShape(),
                        new Shape(256, 384, 3, 3));
                Assert.assertEquals(
                        parameters.get(10).getValue().getArray().getShape(), new Shape(4096, 6400));
                Assert.assertEquals(
                        parameters.get(12).getValue().getArray().getShape(), new Shape(4096, 4096));
            }
        }
    }

    @Test
    public void testTrainWithCustomChannels() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2));
        Block alexNet =
                AlexNet.builder()
                        .setDropOutRate(0.8f)
                        .setNumChannels(new int[] {128, 128, 128, 512, 384, 2048, 2048})
                        .build();
        try (Model model = Model.newInstance("alexnet")) {
            model.setBlock(alexNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 1, 224, 224);
                NDManager manager = trainer.getManager();
                trainer.initialize(inputShape);

                NDArray input = manager.ones(inputShape);
                NDArray label = manager.ones(new Shape(batchSize, 1));
                Batch batch =
                        new Batch(
                                manager.newSubManager(),
                                new NDList(input),
                                new NDList(label),
                                batchSize,
                                Batchifier.STACK,
                                Batchifier.STACK,
                                0,
                                0);
                PairList<String, Parameter> parameters = alexNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(),
                        new Shape(128, 1, 11, 11));
                Assert.assertEquals(
                        parameters.get(2).getValue().getArray().getShape(),
                        new Shape(128, 128, 5, 5));
                Assert.assertEquals(
                        parameters.get(4).getValue().getArray().getShape(),
                        new Shape(128, 128, 3, 3));
                Assert.assertEquals(
                        parameters.get(6).getValue().getArray().getShape(),
                        new Shape(512, 128, 3, 3));
                Assert.assertEquals(
                        parameters.get(8).getValue().getArray().getShape(),
                        new Shape(384, 512, 3, 3));
                Assert.assertEquals(
                        parameters.get(10).getValue().getArray().getShape(), new Shape(2048, 9600));
                Assert.assertEquals(
                        parameters.get(12).getValue().getArray().getShape(), new Shape(2048, 2048));
            }
        }
    }

    @Test
    public void testOutputShapes() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 2;
            NDArray x = manager.ones(new Shape(batchSize, 1, 224, 224));
            Shape currentShape = x.getShape();

            Block alexNet = AlexNet.builder().build();
            alexNet.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            alexNet.initialize(manager, DataType.FLOAT32, currentShape);

            Map<String, Shape> shapeMap = new ConcurrentHashMap<>();
            for (int i = 0; i < alexNet.getChildren().size(); i++) {

                Shape[] newShape =
                        alexNet.getChildren()
                                .get(i)
                                .getValue()
                                .getOutputShapes(new Shape[] {currentShape});
                currentShape = newShape[0];
                shapeMap.put(alexNet.getChildren().get(i).getKey(), currentShape);
            }

            Assert.assertEquals(shapeMap.get("01Conv2d"), new Shape(batchSize, 96, 54, 54));
            Assert.assertEquals(shapeMap.get("04Conv2d"), new Shape(batchSize, 256, 26, 26));
            Assert.assertEquals(shapeMap.get("07Conv2d"), new Shape(batchSize, 384, 12, 12));
            Assert.assertEquals(shapeMap.get("13LambdaBlock"), new Shape(batchSize, 256, 5, 5));
            Assert.assertEquals(shapeMap.get("17Dropout"), new Shape(batchSize, 4096));
        }
    }

    @Test
    public void testForwardMethod() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block alexNet = AlexNet.builder().build();
            int batchSize = 1;
            NDArray x = manager.ones(new Shape(batchSize, 1, 224, 224));
            alexNet.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            alexNet.initialize(manager, DataType.FLOAT32, x.getShape());
            ParameterStore ps = new ParameterStore(manager, true);
            NDArray xHat = alexNet.forward(ps, new NDList(x), false).singletonOrThrow();

            Assert.assertEquals(xHat.getShape(), new Shape(batchSize, 10));
        }
    }
}
