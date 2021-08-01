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
import ai.djl.basicmodelzoo.cv.classification.SqueezeNet;
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

public class SqueezenetTest {

    @Test
    public void testTrain() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2))
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        Block squeezeNet = SqueezeNet.squeezenet(10);
        try (Model model = Model.newInstance("squeezenet")) {
            model.setBlock(squeezeNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 1, 28, 28);
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
                PairList<String, Parameter> parameters = squeezeNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
                NDArray expectedAtIndex0 = manager.ones(new Shape(64, 1, 3, 3));
                NDArray expectedAtIndex1 = manager.zeros(new Shape(64));
                NDArray expectedAtIndex50 = manager.ones(new Shape(10, 512, 1, 1));
                Assertions.assertAlmostEquals(
                        parameters.get(0).getValue().getArray(), expectedAtIndex0);
                Assertions.assertAlmostEquals(
                        parameters.get(1).getValue().getArray(), expectedAtIndex1);
                Assertions.assertAlmostEquals(
                        parameters.get(50).getValue().getArray(), expectedAtIndex50);
            }
        }
    }
}
