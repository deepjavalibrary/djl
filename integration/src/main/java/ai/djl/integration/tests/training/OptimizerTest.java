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

import ai.djl.Device;
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
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.translate.Batchifier;
import org.testng.annotations.Test;

public class OptimizerTest {

    private static final int BATCH_SIZE = 10;
    private static final int CHANNELS = 10;

    @Test
    public void testSgd() {
        Optimizer sgd =
                Optimizer.sgd()
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .build();

        Device[] devices = Device.getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES)
                        .optOptimizer(sgd)
                        .optDevices(devices);
        Block block = Linear.builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance(devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);
                Assertions.assertAlmostEquals(result, manager.create(new float[] {0.68f, -0.16f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.4912f, -0.2544f}));
            }
        }
    }

    @Test
    public void testSgdWithMomentum() {
        Optimizer optim =
                Optimizer.sgd()
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .optMomentum(0.9f)
                        .build();

        Device[] devices = Device.getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance(devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();

                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);
                NDArray result3 = runOptimizer(manager, trainer, block, batchSize);
                for (int i = 0; i < 9; i++) {
                    result3 = runOptimizer(manager, trainer, block, batchSize);
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
                Optimizer.nag()
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .setMomentum(0.9f)
                        .build();

        // Limit to 1 GPU for consist result.
        Device[] devices = Device.getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance(devices[0])) {
            model.setBlock(block);

            int batchSize = config.getDevices().length * BATCH_SIZE;
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);
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
                Optimizer.adam()
                        .optLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .build();

        Device[] devices = Device.getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance(devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);
                Assertions.assertAlmostEquals(
                        result, manager.create(new float[] {0.8999999761581421f, -0.10000064f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.80060977f, -0.19939029f}));
            }
        }
    }

    private NDArray runOptimizer(NDManager manager, Trainer trainer, Block block, int batchSize) {
        NDArray data = manager.ones(new Shape(batchSize, CHANNELS)).mul(2);
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
