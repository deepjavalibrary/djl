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

import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.convolutional.Conv2D;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.nn.norm.BatchNorm;
import software.amazon.ai.training.initializer.Initializer;

public class BlockCoreTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", BlockCoreTest.class.getName()};
        new IntegrationTest().runTests(cmd);
    }

    @RunAsTest
    public void testLinear() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            long outSize = 3;

            Linear linearWithBias = new Linear.Builder().setOutChannels(outSize).build();
            linearWithBias.setInitializer(manager, Initializer.ONES);
            NDArray outBias = linearWithBias.forward(input);
            NDArray expectedBias =
                    input.mmul(manager.ones(new Shape(outSize, 2)).transpose())
                            .add(manager.ones(new Shape(2, outSize)));
            Assertions.assertEquals(expectedBias, outBias);

            Linear linearWithoutBias =
                    new Linear.Builder().setOutChannels(outSize).setBias(false).build();
            linearWithoutBias.setInitializer(manager, Initializer.ONES);
            NDArray outNoBias = linearWithoutBias.forward(input);
            NDArray expectedNoBias = input.mmul(manager.ones(new Shape(outSize, 2)).transpose());
            Assertions.assertEquals(expectedNoBias, outNoBias);
        }
    }

    @RunAsTest
    public void testLinearWithDefinedLayout() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input =
                    manager.create(
                            new float[] {1, 2, 3, 4},
                            new Shape(
                                    new long[] {2, 2},
                                    new LayoutType[] {LayoutType.BATCH, LayoutType.CHANNEL}));
            long outSize = 3;

            Linear linearWithBias = new Linear.Builder().setOutChannels(outSize).build();
            linearWithBias.setInitializer(manager, Initializer.ONES);
            NDArray outBias = linearWithBias.forward(input);
            NDArray expectedBias =
                    input.mmul(manager.ones(new Shape(outSize, 2)).transpose())
                            .add(manager.ones(new Shape(2, outSize)));
            Assertions.assertEquals(expectedBias, outBias);

            Linear linearWithoutBias =
                    new Linear.Builder().setOutChannels(outSize).setBias(false).build();
            linearWithoutBias.setInitializer(manager, Initializer.ONES);
            NDArray outNoBias = linearWithoutBias.forward(input);
            NDArray expectedNoBias = input.mmul(manager.ones(new Shape(outSize, 2)).transpose());
            Assertions.assertEquals(expectedNoBias, outNoBias);
        }
    }

    @RunAsTest
    public void testBatchNorm() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {0, 1, 2, 3}, new Shape(2, 2));
            BatchNorm bn = new BatchNorm.Builder().setAxis(1).build();
            bn.setInitializer(manager, Initializer.ONES);
            NDArray out = bn.forward(input);
            Assertions.assertAlmostEquals(expected, out);
        }
    }

    @RunAsTest
    public void testConv2D() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray expected =
                    manager.create(
                            new float[] {23, 25, 26, 22, 27, 24, 40, 32, 20},
                            new Shape(1, 1, 3, 3));
            Conv2D bn =
                    (Conv2D)
                            new Conv2D.Builder()
                                    .setKernel(new Shape(2, 2))
                                    .setNumFilters(1)
                                    .build();
            bn.setInitializer(manager, Initializer.ONES);
            NDArray out = bn.forward(input);
            Assertions.assertAlmostEquals(expected, out);
        }
    }
}
