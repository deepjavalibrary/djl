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
import ai.djl.basicmodelzoo.cv.classification.GoogLeNet;
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

public class GoogLeNetTest {

    @Test
    public void testTrainWithDefaultChannels() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2));
        Block googLeNet = GoogLeNet.builder().build();
        try (Model model = Model.newInstance("googlenet")) {
            model.setBlock(googLeNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 1, 96, 96);
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
                PairList<String, Parameter> parameters = googLeNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(), new Shape(64, 1, 7, 7));
                Assert.assertEquals(
                        parameters.get(18).getValue().getArray().getShape(),
                        new Shape(128, 256, 1, 1));
                Assert.assertEquals(
                        parameters.get(34).getValue().getArray().getShape(),
                        new Shape(208, 96, 3, 3));
                Assert.assertEquals(
                        parameters.get(60).getValue().getArray().getShape(),
                        new Shape(24, 512, 1, 1));
                Assert.assertEquals(
                        parameters.get(78).getValue().getArray().getShape(),
                        new Shape(256, 528, 1, 1));
                Assert.assertEquals(
                        parameters.get(100).getValue().getArray().getShape(),
                        new Shape(128, 832, 1, 1));
                Assert.assertEquals(
                        parameters.get(114).getValue().getArray().getShape(), new Shape(10, 1024));
            }
        }
    }

    @Test
    public void testOutputShapes() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 1;
            NDArray x = manager.ones(new Shape(batchSize, 1, 96, 96));
            Shape currentShape = x.getShape();

            Block googLeNet = GoogLeNet.builder().build();
            googLeNet.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            googLeNet.initialize(manager, DataType.FLOAT32, currentShape);

            Map<String, Shape> shapeMap = new ConcurrentHashMap<>();
            for (int i = 0; i < googLeNet.getChildren().size(); i++) {
                Shape[] newShape =
                        googLeNet
                                .getChildren()
                                .get(i)
                                .getValue()
                                .getOutputShapes(new Shape[] {currentShape});
                currentShape = newShape[0];
                shapeMap.put(googLeNet.getChildren().get(i).getKey(), currentShape);
            }

            Assert.assertEquals(
                    shapeMap.get("01SequentialBlock"), new Shape(batchSize, 64, 24, 24));
            Assert.assertEquals(
                    shapeMap.get("02SequentialBlock"), new Shape(batchSize, 192, 12, 12));
            Assert.assertEquals(shapeMap.get("03SequentialBlock"), new Shape(batchSize, 480, 6, 6));
            Assert.assertEquals(shapeMap.get("04SequentialBlock"), new Shape(batchSize, 832, 3, 3));
            Assert.assertEquals(shapeMap.get("05SequentialBlock"), new Shape(batchSize, 1024));
        }
    }

    @Test
    public void testForwardMethod() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block googLeNet = GoogLeNet.builder().build();
            int batchSize = 1;
            NDArray x = manager.ones(new Shape(batchSize, 1, 28, 28));
            googLeNet.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            googLeNet.initialize(manager, DataType.FLOAT32, x.getShape());
            ParameterStore ps = new ParameterStore(manager, true);
            NDArray xHat = googLeNet.forward(ps, new NDList(x), true).singletonOrThrow();

            Assert.assertEquals(xHat.getShape(), new Shape(batchSize, 10));
        }
    }
}
