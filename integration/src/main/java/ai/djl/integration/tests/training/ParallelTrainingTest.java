/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.integration.util.TestUtils;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.TrainingListenerAdapter;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ParallelTrainingTest {

    @Test
    public void testParallelUpdatesMatchSequential() {
        TestUtils.requiresEngine("MXNet", "PyTorch");
        Device[] devices = TestUtils.getDevices(2);
        if (!devices[0].isGpu()) {
            throw new SkipException("Training on two devices requires at least one GPU");
        }
        if (devices.length == 1) {
            devices = new Device[] {Device.cpu(), devices[0]};
        }

        float[][] sequential = train(devices, null);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            float[][] parallel = train(devices, executor);
            // Each split contributes to just one weight, so both updates must be present.
            Assert.assertEquals(sequential[0], new float[] {0.4f, 0.8f}, 1e-6f);
            for (int i = 0; i < sequential.length; ++i) {
                Assert.assertEquals(parallel[i], sequential[i], 1e-6f);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    private static float[][] train(Device[] devices, ExecutorService executor) {
        Set<Device> expectedDevices = new HashSet<>(Arrays.asList(devices));
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(devices)
                        .optInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT)
                        .optOptimizer(
                                Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.1f)).build())
                        .addTrainingListeners(
                                new TrainingListenerAdapter() {
                                    @Override
                                    public void onTrainingBatch(Trainer trainer, BatchData data) {
                                        Assert.assertEquals(
                                                data.getLabels().keySet(), expectedDevices);
                                        Assert.assertEquals(
                                                data.getPredictions().keySet(), expectedDevices);
                                    }
                                });
        if (executor != null) {
            config.optExecutorService(executor);
        }

        try (Model model = Model.newInstance("linear", devices[0], TestUtils.getEngine())) {
            Linear block = Linear.builder().setUnits(1).optBias(false).build();
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(4, 2));
                float[][] weights = new float[3][];
                for (int i = 0; i < weights.length; ++i) {
                    NDManager manager = trainer.getManager().newSubManager();
                    try (Batch batch =
                            new Batch(
                                    manager,
                                    new NDList(
                                            manager.create(
                                                    new float[][] {
                                                        {1, 0}, {3, 0}, {0, 1}, {0, 3}
                                                    })),
                                    new NDList(manager.create(new float[][] {{2}, {2}, {4}, {4}})),
                                    4,
                                    Batchifier.STACK,
                                    Batchifier.STACK,
                                    0,
                                    4)) {
                        EasyTrain.trainBatch(trainer, batch);
                        trainer.step();
                        weights[i] = block.getParameters().get("weight").getArray().toFloatArray();
                    }
                }
                return weights;
            }
        }
    }
}
