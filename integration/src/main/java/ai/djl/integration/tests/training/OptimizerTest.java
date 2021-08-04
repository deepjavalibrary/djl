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
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import org.testng.annotations.Test;

public class OptimizerTest {

    private static final int BATCH_SIZE = 10;
    private static final int CHANNELS = 10;

    @Test
    public void testSgd() {
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.1f)).build();

        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(sgd)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
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
                        .setLearningRateTracker(Tracker.fixed(0.1f))
                        .optMomentum(0.9f)
                        .build();

        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
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
                        .setLearningRateTracker(Tracker.fixed(0.1f))
                        .setMomentum(0.9f)
                        .build();

        // Limit to 1 GPU for consist result.
        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
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
        Optimizer optim = Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.1f)).build();

        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
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

    @Test
    public void testAdagrad() {
        Optimizer optim = Optimizer.adagrad().optLearningRateTracker(Tracker.fixed(0.1f)).build();

        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);

                Assertions.assertAlmostEquals(result, manager.create(new float[] {0.9f, -0.1f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.834f, -0.1656f}));
            }
        }
    }

    @Test
    public void testRMSProp() {
        Optimizer optim =
                Optimizer.rmsprop()
                        .optLearningRateTracker(Tracker.fixed(0.1f))
                        .optCentered(false)
                        .build();

        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);

                Assertions.assertAlmostEquals(
                        result, manager.create(new float[] {0.6838f, -0.3162f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.5178f, -0.4822f}));
            }
        }
    }

    @Test
    public void testRMSPropAlex() {
        Optimizer optim =
                Optimizer.rmsprop()
                        .optLearningRateTracker(Tracker.fixed(0.1f))
                        .optCentered(true)
                        .build();

        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);

                Assertions.assertAlmostEquals(
                        result, manager.create(new float[] {0.6667f, -0.3333f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.189f, -0.811f}));
            }
        }
    }

    @Test
    public void testAdadelta() {
        Optimizer optim = Optimizer.adadelta().build();

        Device[] devices = Engine.getInstance().getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optim)
                        .optDevices(devices);

        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);

                Assertions.assertAlmostEquals(result, manager.create(new float[] {0.999f, 0f}));
                Assertions.assertAlmostEquals(result2, manager.create(new float[] {0.999f, 0f}));
            }
        }
    }

    private NDArray runOptimizer(NDManager manager, Trainer trainer, Block block, int batchSize) {
        NDArray data = manager.ones(new Shape(batchSize, CHANNELS)).mul(2);
        NDArray label = data.mul(2);
        Batch batch =
                new Batch(
                        manager.newSubManager(),
                        new NDList(data),
                        new NDList(label),
                        batchSize,
                        Batchifier.STACK,
                        Batchifier.STACK,
                        0,
                        0);
        EasyTrain.trainBatch(trainer, batch);
        trainer.step();
        return NDArrays.stack(
                new NDList(
                        block.getParameters()
                                .stream()
                                .map(paramPair -> paramPair.getValue().getArray().mean())
                                .toArray(NDArray[]::new)));
    }
}
