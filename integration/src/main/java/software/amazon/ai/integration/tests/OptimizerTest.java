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

import java.util.Arrays;
import java.util.stream.Stream;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.TrainingController;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.optimizer.Adam;
import software.amazon.ai.training.optimizer.Nag;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.learningrate.LrTracker;

public class OptimizerTest {

    private static final int BATCH_SIZE = 10;
    private static final int CHANNELS = 10;

    public static void main(String[] args) {
        String[] cmd = {"-c", OptimizerTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testSgd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = block(manager);
            Optimizer optim =
                    new Sgd.Builder()
                            .setRescaleGrad(1.0f / BATCH_SIZE)
                            .setLrTracker(LrTracker.fixedLR(1E7f))
                            .build();
            NDArray result = runOptimizer(manager, block, optim);
            NDArray result2 = runOptimizer(manager, block, optim);
            Assertions.assertAlmostEquals(manager.create(new float[] {0.9963f, 1.0231f}), result);
            Assertions.assertAlmostEquals(manager.create(new float[] {1.0222f, 1.0625f}), result2);
        }
    }

    @RunAsTest
    public void testSgdWithMomentum() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = block(manager);
            Optimizer optim =
                    new Sgd.Builder()
                            .setRescaleGrad(1.0f / BATCH_SIZE)
                            .setLrTracker(LrTracker.fixedLR(1E7f))
                            .optMomentum(1E2f)
                            .build();
            NDArray result = runOptimizer(manager, block, optim);
            NDArray result2 = runOptimizer(manager, block, optim);
            Assertions.assertAlmostEquals(manager.create(new float[] {0.9963f, 1.0231f}), result);
            Assertions.assertAlmostEquals(manager.create(new float[] {0.6516f, 3.3688f}), result2);
        }
    }

    @RunAsTest
    public void testNag() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = block(manager);
            Optimizer optim =
                    new Nag.Builder()
                            .setRescaleGrad(1.0f / BATCH_SIZE)
                            .setLrTracker(LrTracker.fixedLR(1E7f))
                            .setMomentum(1E1f)
                            .build();
            NDArray result = runOptimizer(manager, block, optim);
            NDArray result2 = runOptimizer(manager, block, optim);
            Assertions.assertAlmostEquals(manager.create(new float[] {0.959f, 1.2541f}), result);
            Assertions.assertAlmostEquals(manager.create(new float[] {-0.61f, 5f}), result2);
        }
    }

    @RunAsTest
    public void testAdam() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = block(manager);
            Optimizer optim =
                    new Adam.Builder()
                            .setRescaleGrad(1.0f / BATCH_SIZE)
                            .optLearningRate(1E2f)
                            .build();
            NDArray result = runOptimizer(manager, block, optim);
            NDArray result2 = runOptimizer(manager, block, optim);
            Assertions.assertAlmostEquals(manager.create(new float[] {0.8849f, 1.7222f}), result);
            Assertions.assertAlmostEquals(
                    manager.create(new float[] {60.4156f, 61.2529f}), result2);
        }
    }

    private Block block(NDManager manager) {
        Linear linear = new Linear.Builder().setOutChannels(CHANNELS).build();
        linear.setInitializer(manager, Initializer.ONES, true);
        return linear;
    }

    private NDArray runOptimizer(NDManager manager, Block block, Optimizer optim) {
        NDArray data = manager.ones(new Shape(BATCH_SIZE, CHANNELS));
        NDArray label = manager.arange(0, BATCH_SIZE);
        TrainingController controller = new TrainingController(block.getParameters(), optim);
        try (GradientCollector gradCol = GradientCollector.newInstance()) {
            NDArray pred = block.forward(new NDList(data)).head();
            NDArray loss = Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
            gradCol.backward(loss);
        }
        controller.step();
        return NDArrays.stack(
                block.getParameters()
                        .stream()
                        .map(paramPair -> paramPair.getValue().getArray().mean())
                        .toArray(NDArray[]::new));
    }
}
