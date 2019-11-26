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
import ai.djl.integration.util.Assertions;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Nag;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.translate.Batchifier;
import org.testng.annotations.Test;

public class OptimizerTest {

    private static final int BATCH_SIZE = 10;
    private static final int CHANNELS = 10;

    @Test
    public void testSgd() {
        Optimizer sgd =
                new Sgd.Builder()
                        .setRescaleGrad(1.0f / BATCH_SIZE)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .build();

        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss()).setOptimizer(sgd);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(BATCH_SIZE, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                Assertions.assertAlmostEquals(result, manager.create(new float[] {0.68f, -0.16f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.4912f, -0.2544f}));
            }
        }
    }

    @Test
    public void testSgdWithMomentum() {
        Optimizer optim =
                new Sgd.Builder()
                        .setRescaleGrad(1.0f / BATCH_SIZE)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .optMomentum(0.9f)
                        .build();

        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss()).setOptimizer(optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(BATCH_SIZE, CHANNELS));

                NDManager manager = trainer.getManager();

                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                NDArray result3 = runOptimizer(manager, trainer, block);
                for (int i = 0; i < 9; i++) {
                    result3 = runOptimizer(manager, trainer, block);
                }
                Assertions.assertAlmostEquals(result, manager.create(new float[] {0.68f, -0.16f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.2032f, -0.3984f}));
                Assertions.assertAlmostEquals(
                        result3, manager.create(new float[] {0.04637566f, -0.4768118f}));
            }
        }
    }

    @Test
    public void testNag() {
        Optimizer optim =
                new Nag.Builder()
                        .setRescaleGrad(1.0f / BATCH_SIZE)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .setMomentum(0.9f)
                        .build();

        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss()).setOptimizer(optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(BATCH_SIZE, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                Assertions.assertAlmostEquals(
                        result, manager.create(new float[] {0.392f, -0.304f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {-0.0016f, -0.5008f}));
            }
        }
    }

    @Test
    public void testAdam() {
        Optimizer optim =
                new Adam.Builder()
                        .setRescaleGrad(1.0f / BATCH_SIZE)
                        .optLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .build();

        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss()).setOptimizer(optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(BATCH_SIZE, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                Assertions.assertAlmostEquals(
                        result, manager.create(new float[] {0.8999999761581421f, -0.10000064f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.80060977f, -0.19939029f}));
            }
        }
    }

    private NDArray runOptimizer(NDManager manager, Trainer trainer, Block block) {
        NDArray data = manager.ones(new Shape(BATCH_SIZE, CHANNELS)).mul(2);
        NDArray label = data.mul(2);
        Batch batch = new Batch(manager, new NDList(data), new NDList(label), Batchifier.STACK);
        trainer.trainBatch(batch);
        trainer.step();
        return NDArrays.stack(
                new NDList(
                        block.getParameters()
                                .stream()
                                .map(paramPair -> paramPair.getValue().getArray().mean())
                                .toArray(NDArray[]::new)));
    }
}
