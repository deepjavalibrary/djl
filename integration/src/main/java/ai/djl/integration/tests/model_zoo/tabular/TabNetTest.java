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
package ai.djl.integration.tests.model_zoo.tabular;

import ai.djl.Model;
import ai.djl.basicmodelzoo.tabular.TabNet;
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
import ai.djl.training.loss.TabNetRegressionLoss;
import ai.djl.translate.Batchifier;
import ai.djl.util.PairList;

import org.testng.Assert;
import org.testng.annotations.Test;

public class TabNetTest {
    @Test
    public void testTabNetGLU() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        try (Model model = Model.newInstance("model")) {
            model.setBlock(TabNet.tabNetGLUBlock(1));

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(4));
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 2, 3, 4});
                data = data.reshape(2, 2);
                // expected calculated through pytorch
                NDArray expected = manager.create(new float[] {0.8808f, 2.946f});
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow().squeeze();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testTrainingAndLogic() {
        TrainingConfig config =
                new DefaultTrainingConfig(new TabNetRegressionLoss())
                        .optDevices(Engine.getInstance().getDevices(2));

        Block tabNet = TabNet.builder().setOutDim(10).build();
        try (Model model = Model.newInstance("tabNet")) {
            model.setBlock(tabNet);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 32;
                Shape inputShape = new Shape(batchSize, 128);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray input = manager.randomUniform(0, 1, inputShape);
                NDArray label = manager.ones(new Shape(batchSize, 10));
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
                PairList<String, Parameter> parameters = tabNet.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
                // the gamma of batchNorm Layer
                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(), new Shape(128));

                // weight of shared fullyConnected Block0
                Assert.assertEquals(
                        parameters.get(4).getValue().getArray().getShape(), new Shape(256, 128));

                // the parameter value of a shared fc Block should be the same
                Assert.assertEquals(parameters.get(8).getValue(), parameters.get(4).getValue());
                Assert.assertEquals(parameters.get(32).getValue(), parameters.get(4).getValue());

                // fc's weight of attention Transformer of step01
                Assert.assertEquals(
                        parameters.get(56).getValue().getArray().getShape(), new Shape(128, 64));

                // the final fc Block
                Assert.assertEquals(
                        parameters.get(152).getValue().getArray().getShape(), new Shape(10, 64));
            }
        }
    }
}
