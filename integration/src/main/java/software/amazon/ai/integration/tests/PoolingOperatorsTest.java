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
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.pooling.Pool;

public class PoolingOperatorsTest {

    public static void main(String[] args) {
        String[] cmd = {"-c", PoolingOperatorsTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testMaxPool() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray maxPoolActual =
                    Pool.maxPool(original, new Shape(2, 2), new Shape(1, 1), new Shape(0, 0));
            NDArray maxPoolExpected =
                    manager.create(
                            new float[] {9, 9, 9, 11, 11, 9, 13, 11, 8}, new Shape(1, 1, 3, 3));
            Assertions.assertEquals(maxPoolExpected, maxPoolActual, "Max Pooling operation failed");
        }
    }

    @RunAsTest
    public void testSumPool() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray maxPoolActual =
                    Pool.sumPool(original, new Shape(2, 2), new Shape(1, 1), new Shape(0, 0));
            NDArray maxPoolExpected =
                    manager.create(
                            new float[] {22, 24, 25, 21, 26, 23, 39, 31, 19},
                            new Shape(1, 1, 3, 3));
            Assertions.assertEquals(maxPoolExpected, maxPoolActual, "Sum Pooling operation failed");
        }
    }

    @RunAsTest
    public void testAvgPool() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray maxPoolActual =
                    Pool.avgPool(original, new Shape(2, 2), new Shape(2, 2), new Shape(0, 0));
            NDArray maxPoolExpected =
                    manager.create(new float[] {5.5f, 6.25f, 9.75f, 4.75f}, new Shape(1, 1, 2, 2));
            Assertions.assertEquals(maxPoolExpected, maxPoolActual, "Avg Pooling operation failed");
        }
    }

    @RunAsTest
    public void testLpPool() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray maxPoolActual =
                    Pool.lpPool(original, new Shape(2, 2), new Shape(1, 1), new Shape(0, 0), 1);
            NDArray maxPoolExpected =
                    manager.create(
                            new float[] {22, 24, 25, 21, 26, 23, 39, 31, 19},
                            new Shape(1, 1, 3, 3));
            Assertions.assertEquals(maxPoolExpected, maxPoolActual, "Sum Pooling operation failed");
        }
    }
}
