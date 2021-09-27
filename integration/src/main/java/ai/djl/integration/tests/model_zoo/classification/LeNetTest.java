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
import ai.djl.basicmodelzoo.cv.classification.LeNet;
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

public class LeNetTest {

    @Test
    public void testTrainWithDefaultChannels() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2));
        Block leNet = LeNet.builder().build();
        try (Model model = Model.newInstance("lenet")) {
            model.setBlock(leNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 1, 28, 28);
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
                PairList<String, Parameter> parameters = leNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(), new Shape(6, 1, 5, 5));
                Assert.assertEquals(
                        parameters.get(1).getValue().getArray().getShape(), new Shape(16, 6, 5, 5));
                Assert.assertEquals(
                        parameters.get(3).getValue().getArray().getShape(), new Shape(120, 400));
                Assert.assertEquals(
                        parameters.get(5).getValue().getArray().getShape(), new Shape(84, 120));
                Assert.assertEquals(
                        parameters.get(8).getValue().getArray().getShape(), new Shape(10));
            }
        }
    }

    @Test
    public void testTrainWithCustomChannels() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2));
        Block leNet = LeNet.builder().setNumChannels(new int[] {12, 15, 150, 100}).build();
        try (Model model = Model.newInstance("lenet")) {
            model.setBlock(leNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 1, 28, 28);
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
                PairList<String, Parameter> parameters = leNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(), new Shape(12, 1, 5, 5));
                Assert.assertEquals(
                        parameters.get(1).getValue().getArray().getShape(),
                        new Shape(15, 12, 5, 5));
                Assert.assertEquals(
                        parameters.get(3).getValue().getArray().getShape(), new Shape(150, 375));
                Assert.assertEquals(
                        parameters.get(5).getValue().getArray().getShape(), new Shape(100, 150));
                Assert.assertEquals(
                        parameters.get(8).getValue().getArray().getShape(), new Shape(10));
            }
        }
    }

    @Test
    public void testOutputShapes() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 1;
            NDArray x = manager.ones(new Shape(batchSize, 1, 28, 28));
            Shape currentShape = x.getShape();

            Block leNet = LeNet.builder().build();
            leNet.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            leNet.initialize(manager, DataType.FLOAT32, currentShape);

            Map<String, Shape> shapeMap = new ConcurrentHashMap<>();
            for (int i = 0; i < leNet.getChildren().size(); i++) {
                Shape[] newShape =
                        leNet.getChildren()
                                .get(i)
                                .getValue()
                                .getOutputShapes(new Shape[] {currentShape});
                currentShape = newShape[0];
                shapeMap.put(leNet.getChildren().get(i).getKey(), currentShape);
            }

            Assert.assertEquals(shapeMap.get("01Conv2d"), new Shape(batchSize, 6, 28, 28));
            Assert.assertEquals(shapeMap.get("04Conv2d"), new Shape(batchSize, 16, 10, 10));
            Assert.assertEquals(shapeMap.get("08Linear"), new Shape(batchSize, 120));
            Assert.assertEquals(shapeMap.get("12Linear"), new Shape(batchSize, 10));
        }
    }

    @Test
    public void testForwardMethod() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block leNet = LeNet.builder().build();
            int batchSize = 1;
            NDArray x = manager.ones(new Shape(batchSize, 1, 28, 28));
            leNet.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            leNet.initialize(manager, DataType.FLOAT32, x.getShape());
            ParameterStore ps = new ParameterStore(manager, true);
            NDArray xHat = leNet.forward(ps, new NDList(x), true).singletonOrThrow();

            Assert.assertEquals(xHat.getShape(), new Shape(batchSize, 10));
        }
    }
}
