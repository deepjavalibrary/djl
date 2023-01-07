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
package ai.djl.integration.tests.training;

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.integration.util.TestUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.testing.Assertions;
import ai.djl.testing.TestRequirements;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class GradientCollectorIntegrationTest {

    @Test
    public void testAutograd() {
        try (Model model = Model.newInstance("model", TestUtils.getEngine());
                NDManager manager = model.getNDManager()) {
            model.setBlock(Blocks.identityBlock());
            try (Trainer trainer =
                    model.newTrainer(
                            new DefaultTrainingConfig(Loss.l2Loss())
                                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT))) {
                try (GradientCollector gradCol = trainer.newGradientCollector()) {
                    NDArray lhs =
                            manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
                    NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
                    NDArray expected =
                            manager.create(new float[] {2, 3, -4, 2, 3, -4}, new Shape(2, 3));
                    lhs.setRequiresGradient(true);
                    // autograd automatically set recording and training during initialization

                    NDArray result = NDArrays.dot(lhs, rhs);
                    gradCol.backward(result);
                    NDArray grad = lhs.getGradient();
                    Assertions.assertAlmostEquals(grad, expected);
                    // test close and get again
                    grad.close();
                    NDArray grad2 = lhs.getGradient();
                    Assertions.assertAlmostEquals(grad2, expected);
                }
            }
        }
    }

    /** Tests that the gradients do not accumulate when closing the gradient collector. */
    @Test
    public void testClearGradients() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray variable = manager.create(0.0f);
            variable.setRequiresGradient(true);

            Engine engine = manager.getEngine();
            for (int i = 0; i < 3; i++) {
                manager.zeroGradients();
                try (GradientCollector gc = engine.newGradientCollector()) {
                    NDArray loss = variable.mul(2);
                    gc.backward(loss);
                }
                Assert.assertEquals(variable.getGradient().getFloat(), 2.0f);
            }
        }
    }

    @Test
    public void testFreezeParameters() {
        try (Model model = Model.newInstance("model", TestUtils.getEngine())) {
            Block blockFrozen = new Mlp(10, 10, new int[] {10});
            Block blockNormal = new Mlp(10, 10, new int[] {10});
            Block combined = new SequentialBlock().add(blockFrozen).add(blockNormal);
            model.setBlock(combined);

            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss())
                            .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 10));

                blockFrozen.freezeParameters(true);

                // Find total params
                Float frozenVal =
                        blockFrozen.getParameters().valueAt(0).getArray().sum().getFloat();
                Float normalVal =
                        blockNormal.getParameters().valueAt(0).getArray().sum().getFloat();

                // Run training step
                NDManager manager = trainer.getManager();
                NDArray data = manager.arange(100.0f).reshape(new Shape(10, 10));
                NDArray labels = manager.arange(100.0f).reshape(new Shape(10, 10));
                Batch batch =
                        new Batch(
                                manager,
                                new NDList(data),
                                new NDList(labels),
                                1,
                                Batchifier.STACK,
                                Batchifier.STACK,
                                0,
                                1);
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                // Check updated total params
                // The frozen one should not have changed, but normal one should
                Float newFrozenVal =
                        blockFrozen.getParameters().valueAt(0).getArray().sum().getFloat();
                Float newNormalVal =
                        blockNormal.getParameters().valueAt(0).getArray().sum().getFloat();
                Assert.assertEquals(newFrozenVal, frozenVal);
                Assert.assertNotEquals(newNormalVal, normalVal);

                blockFrozen.freezeParameters(false);

                // Check that unfreezing the block now makes it update
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Float nowUnfrozenVal =
                        blockFrozen.getParameters().valueAt(0).getArray().sum().getFloat();
                Assert.assertNotEquals(nowUnfrozenVal, frozenVal);
            }
        }
    }

    @Test
    public void testTrain() throws IOException, TranslateException {
        TestRequirements.nightly();

        int numOfData = 1000;
        int batchSize = 10;
        int epochs = 10;

        Optimizer optimizer = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(.03f)).build();

        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .addTrainingListeners(new EvaluatorTrainingListener())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer);

        try (Model model = Model.newInstance("linear", TestUtils.getEngine())) {
            Linear block = Linear.builder().setUnits(1).build();
            model.setBlock(block);

            NDManager manager = model.getNDManager();

            NDArray weight = manager.create(new float[] {2.f, -3.4f}, new Shape(2, 1));
            float bias = 4.2f;
            NDArray data = manager.randomNormal(new Shape(numOfData, weight.size(0)));
            // y = w * x + b
            NDArray label = data.dot(weight).add(bias);
            // add noise
            label.addi(
                    manager.randomNormal(
                            0f, 0.01f, label.getShape(), DataType.FLOAT32, manager.getDevice()));

            int sampling = config.getDevices().length * batchSize;
            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .optLabels(label)
                            .setSampling(sampling, false)
                            .build();
            float lossValue;
            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(sampling, weight.size(0));
                trainer.initialize(inputShape);

                for (int epoch = 0; epoch < epochs; epoch++) {
                    trainer.notifyListeners(listener -> listener.onEpoch(trainer));
                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        EasyTrain.trainBatch(trainer, batch);
                        trainer.step();
                        batch.close();
                    }
                }
                lossValue = trainer.getLoss().getAccumulator(EvaluatorTrainingListener.TRAIN_EPOCH);
            }
            float expectedLoss = 0.001f;
            Assert.assertTrue(
                    lossValue < expectedLoss,
                    String.format(
                            "Loss did not improve, loss value: %f, expected "
                                    + "max loss value: %f",
                            lossValue, expectedLoss));
        }
    }
}
