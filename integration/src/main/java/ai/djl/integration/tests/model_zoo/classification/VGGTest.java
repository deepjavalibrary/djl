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
import ai.djl.basicmodelzoo.cv.classification.VGG;
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
import org.testng.SkipException;
import org.testng.annotations.Test;

public class VGGTest {

    @Test
    public void testTrainWithDefaultChannels() {
        // Vgg is too large to be fit into the github action cpu main memory
        // only run the test when we have GPU
        if (Engine.getInstance().getGpuCount() == 0) {
            throw new SkipException("GPU only");
        }
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2));

        Block vgg = VGG.builder().build();
        try (Model model = Model.newInstance("vgg")) {
            model.setBlock(vgg);
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
                PairList<String, Parameter> parameters = vgg.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(), new Shape(64, 1, 3, 3));
                Assert.assertEquals(
                        parameters.get(2).getValue().getArray().getShape(),
                        new Shape(128, 64, 3, 3));
                Assert.assertEquals(
                        parameters.get(4).getValue().getArray().getShape(),
                        new Shape(256, 128, 3, 3));
                Assert.assertEquals(
                        parameters.get(8).getValue().getArray().getShape(),
                        new Shape(512, 256, 3, 3));
                Assert.assertEquals(
                        parameters.get(12).getValue().getArray().getShape(),
                        new Shape(512, 512, 3, 3));
                Assert.assertEquals(
                        parameters.get(16).getValue().getArray().getShape(),
                        new Shape(4096, 25088));
                Assert.assertEquals(
                        parameters.get(18).getValue().getArray().getShape(), new Shape(4096, 4096));
            }
        }
    }

    @Test
    public void testOutputShapes() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 1;
            NDArray x = manager.ones(new Shape(batchSize, 1, 224, 224));
            Shape currentShape = x.getShape();

            Block vgg = VGG.builder().build();
            vgg.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            vgg.initialize(manager, DataType.FLOAT32, currentShape);

            Map<String, Shape> shapeMap = new ConcurrentHashMap<>();
            for (int i = 0; i < vgg.getChildren().size(); i++) {
                Shape[] newShape =
                        vgg.getChildren()
                                .get(i)
                                .getValue()
                                .getOutputShapes(new Shape[] {currentShape});
                currentShape = newShape[0];
                shapeMap.put(vgg.getChildren().get(i).getKey(), currentShape);
            }

            Assert.assertEquals(
                    shapeMap.get("01SequentialBlock"), new Shape(batchSize, 64, 112, 112));
            Assert.assertEquals(
                    shapeMap.get("02SequentialBlock"), new Shape(batchSize, 128, 56, 56));
            Assert.assertEquals(
                    shapeMap.get("03SequentialBlock"), new Shape(batchSize, 256, 28, 28));
            Assert.assertEquals(
                    shapeMap.get("04SequentialBlock"), new Shape(batchSize, 512, 14, 14));
            Assert.assertEquals(shapeMap.get("05SequentialBlock"), new Shape(batchSize, 512, 7, 7));
            Assert.assertEquals(shapeMap.get("07Linear"), new Shape(batchSize, 4096));
        }
    }

    @Test
    public void testForwardMethod() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block vgg = VGG.builder().build();
            int batchSize = 1;
            NDArray x = manager.ones(new Shape(batchSize, 1, 224, 224));
            vgg.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            vgg.initialize(manager, DataType.FLOAT32, x.getShape());
            ParameterStore ps = new ParameterStore(manager, true);
            NDArray xHat = vgg.forward(ps, new NDList(x), false).singletonOrThrow();

            Assert.assertEquals(xHat.getShape(), new Shape(batchSize, 10));
        }
    }
}
