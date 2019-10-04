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

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true, sgd);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                // TODO: fix atol and rtol too large on GPU build
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {0.6600000262260437f, 0.8300000429153442f}),
                        result);
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {0.4593999981880188f, 0.729699969291687f}),
                        result2);
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

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true, optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                // TODO: fix atol and rtol too large on GPU build
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {0.6600000262260437f, 0.8300000429153442f}),
                        result);
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {0.15339994430541992f, 0.57669997215271f}),
                        result2);
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

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true, optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                // TODO: fix atol and rtol too large on GPU build
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {0.3539999723434448f, 0.6769999861717224f}),
                        result);
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {-0.06416600942611694f, 0.4679170250892639f}),
                        result2);
            }
        }
    }

    @Test
    public void testAdam() {
        Optimizer optim =
                new Adam.Builder().setRescaleGrad(1.0f / BATCH_SIZE).optLearningRate(0.1f).build();

        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true, optim);
        Block block = new Linear.Builder().setOutChannels(CHANNELS).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block);
                NDArray result2 = runOptimizer(manager, trainer, block);
                // TODO: fix atol and rtol too large on GPU build
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {0.8999999761581421f, 0.8999999761581421f}),
                        result);
                Assertions.assertAlmostEquals(
                        manager.create(new float[] {0.8005584478378296f, 0.8005584478378296f}),
                        result2);
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
