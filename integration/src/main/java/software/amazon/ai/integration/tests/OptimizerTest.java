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
package software.amazon.ai.integration.tests;

import org.apache.mxnet.engine.MxGradientCollector;
import org.testng.annotations.Test;
import software.amazon.ai.Model;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.optimizer.Adam;
import software.amazon.ai.training.optimizer.Nag;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.learningrate.LearningRateTracker;

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

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, sgd);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new DataDesc[] {new DataDesc(new Shape(BATCH_SIZE, CHANNELS))});

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                // TODO: fix atol and rtol too large on GPU build
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

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new DataDesc[] {new DataDesc(new Shape(BATCH_SIZE, CHANNELS))});

                NDManager manager = trainer.getManager();

                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                // TODO: fix atol and rtol too large on GPU build
                Assertions.assertAlmostEquals(result, manager.create(new float[] {0.68f, -0.16f}));
                Assertions.assertAlmostEquals(
                        result2, manager.create(new float[] {0.2032f, -0.3984f}));
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

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new DataDesc[] {new DataDesc(new Shape(BATCH_SIZE, CHANNELS))});

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                // TODO: fix atol and rtol too large on GPU build
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
                new Adam.Builder().setRescaleGrad(1.0f / BATCH_SIZE).optLearningRate(0.1f).build();

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new DataDesc[] {new DataDesc(new Shape(BATCH_SIZE, CHANNELS))});

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
        try (GradientCollector gradCol = new MxGradientCollector()) {
            NDArray pred = trainer.forward(new NDList(data)).head();
            NDArray loss = Loss.l2Loss(label, pred);
            gradCol.backward(loss);
        }
        trainer.step();
        return NDArrays.stack(
                new NDList(
                        block.getParameters()
                                .stream()
                                .map(paramPair -> paramPair.getValue().getArray().mean())
                                .toArray(NDArray[]::new)));
    }
}
