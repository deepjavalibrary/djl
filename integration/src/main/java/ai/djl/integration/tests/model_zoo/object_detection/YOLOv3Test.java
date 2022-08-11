/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.integration.tests.model_zoo.object_detection;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.object_detection.yolo.YOLOV3;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Batchifier;
import ai.djl.util.PairList;

import org.testng.annotations.Test;

public class YOLOv3Test {
    @Test
    public void testDarkNet() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2))
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        Block darkNet = YOLOV3.builder().buildDarkNet();
        try (Model model = Model.newInstance("darkNet")) {
            model.setBlock(darkNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 3, 224, 224);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
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
                PairList<String, Parameter> parameters = darkNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                // expected kernelShapes
                NDArray expectedAtIndex0 =
                        manager.ones(new Shape(32, 3, 3, 3)); // 32*3*3*3 for first layer
                NDArray expectedAtIndex6 =
                        manager.ones(new Shape(64, 32, 3, 3)); // 64*32*3*3 for second layer
                NDArray expectedAtIndex120 =
                        manager.ones(
                                new Shape(
                                        128, 256, 1, 1)); // 128*256*1*1  for pointWiseLayer at last
                NDArray expectedLinear =
                        manager.ones(
                                new Shape(10, 1024)); // outSize = 10, AvgPoolSize should be 10*1024

                // test if the shape of the kernel sizes are right
                Assertions.assertAlmostEquals(
                        parameters.get(0).getValue().getArray(), expectedAtIndex0);
                Assertions.assertAlmostEquals(
                        parameters.get(6).getValue().getArray(), expectedAtIndex6);
                Assertions.assertAlmostEquals(
                        parameters.get(120).getValue().getArray(), expectedAtIndex120);
                Assertions.assertAlmostEquals(
                        parameters.get(312).getValue().getArray(), expectedLinear);
            }
        }
    }
}
