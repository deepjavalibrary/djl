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

package ai.djl.integration.tests.model_zoo.classification;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.MobileNetV2;
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

import java.util.Arrays;

public class MobileNetV2Test {
    @Test
    public void testTrain() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2))
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        Block mobilenet = MobileNetV2.builder().setOutSize(10).build();
        try (Model model = Model.newInstance("mobilenet")) {
            model.setBlock(mobilenet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 3, 224, 224);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray input = manager.randomUniform(0, 1, inputShape);
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
                PairList<String, Parameter> parameters = mobilenet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                NDArray expectedAtIndex0 =
                        manager.ones(new Shape(32, 3, 1, 1)); // 32*3*1*1 for first layer
                NDArray expectedAtIndex6 =
                        manager.ones(new Shape(32, 32, 1, 1)); // 32*32*1*1 for pointWiseLayer1
                NDArray expectedAtIndex12 =
                        manager.ones(new Shape(32, 1, 3, 3)); // 32*1*3*3 for depthWiseLayer1
                NDArray expectedAtIndex312 =
                        manager.ones(
                                new Shape(
                                        1280, 320, 1,
                                        1)); // 1280*320*1*1  for pointWiseLayer at last

                Assertions.assertAlmostEquals(
                        parameters.get(0).getValue().getArray(), expectedAtIndex0);
                Assertions.assertAlmostEquals(
                        parameters.get(6).getValue().getArray(), expectedAtIndex6);
                Assertions.assertAlmostEquals(
                        parameters.get(12).getValue().getArray(), expectedAtIndex12);
                Assertions.assertAlmostEquals(
                        parameters.get(312).getValue().getArray(), expectedAtIndex312);
            }
        }
    }

    public static void main(String[] args) {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2)) //use a simple gpu
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        Block mobilenet = MobileNetV2.builder().setOutSize(10).build();

        try(Model model = Model.newInstance("mobilenet")){
            model.setBlock(mobilenet);
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
                PairList<String, Parameter> parameters = mobilenet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
                for(int i = 0;i<parameters.size();i+=5){
                    System.out.println(parameters.get(i).getValue().getArray().getShape());
                }
            }
        }
        System.out.println("---------------------------------------------------");
        getOutputShapes(mobilenet,new Shape[]{new Shape(1,3,224,224)});
    }

    private static void getOutputShapes(Block mobilenet,Shape[] inputs){
        Shape[] current = inputs;
        System.out.println(Arrays.toString(current));
        for(Block block:mobilenet.getChildren().values()){
            current = block.getOutputShapes(current);
            System.out.println(Arrays.toString(current));
        }
    }
}
