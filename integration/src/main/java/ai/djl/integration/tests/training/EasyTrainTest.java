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
import ai.djl.nn.LambdaBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Batchifier;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public class EasyTrainTest {

    @Test
    public void testTrainingSplitFailure() {
        checkSplitFailure(EasyTrain::trainBatch);
    }

    @Test
    public void testValidationSplitFailure() {
        checkSplitFailure(EasyTrain::validateBatch);
    }

    private static void checkSplitFailure(BiConsumer<Trainer, Batch> processBatch) {
        TestUtils.requiresEngine("MXNet", "PyTorch");
        ExecutorService executor = Executors.newSingleThreadExecutor();
        AtomicInteger attempts = new AtomicInteger();
        IllegalStateException failure = new IllegalStateException("failed split");
        try (Model model = Model.newInstance("failing", Device.cpu(), TestUtils.getEngine())) {
            model.setBlock(
                    new LambdaBlock(
                            inputs -> {
                                attempts.incrementAndGet();
                                throw failure;
                            }));
            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss())
                            .optDevices(new Device[] {Device.cpu()})
                            .optExecutorService(executor);
            try (Trainer trainer = model.newTrainer(config);
                    Batch batch = twoCpuSplits(model.getNDManager())) {
                trainer.initialize(new Shape(1));
                CompletionException ex =
                        Assert.expectThrows(
                                CompletionException.class,
                                () -> processBatch.accept(trainer, batch));
                Assert.assertSame(ex.getCause(), failure);
                Assert.assertEquals(attempts.get(), 2);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    private static Batch twoCpuSplits(NDManager manager) {
        NDList data = new NDList(manager.create(new float[] {1f, 2f}));
        NDList labels = new NDList(manager.create(new float[] {1f, 2f}));
        return new Batch(
                manager.newSubManager(),
                data,
                labels,
                2,
                Batchifier.STACK,
                Batchifier.STACK,
                0,
                2) {
            @Override
            public Batch[] split(Device[] devices, boolean evenSplit) {
                // Exercise the executor path on machines with only one device.
                return super.split(new Device[] {Device.cpu(), Device.cpu()}, evenSplit);
            }
        };
    }
}
