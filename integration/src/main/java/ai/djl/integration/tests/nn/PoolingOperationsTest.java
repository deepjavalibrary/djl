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
package ai.djl.integration.tests.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.pooling.Pool;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PoolingOperationsTest {

    @Test
    public void testMaxPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result =
                    Pool.maxPool(original, new Shape(2, 2), new Shape(1, 1), new Shape(0, 0));
            NDArray expected =
                    manager.create(
                            new float[] {9, 9, 9, 11, 11, 9, 13, 11, 8}, new Shape(1, 1, 3, 3));
            Assert.assertEquals(result, expected, "Max Pooling operation failed");
        }
    }

    @Test
    public void testGlobalMaxPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result = Pool.globalMaxPool(original);
            NDArray expected = manager.create(new float[] {13}, new Shape(1, 1, 1, 1));
            Assert.assertEquals(result, expected, "Global Max Pooling operation failed");
        }
    }

    @Test
    public void testSumPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result =
                    Pool.sumPool(original, new Shape(2, 2), new Shape(1, 1), new Shape(0, 0));
            NDArray expected =
                    manager.create(
                            new float[] {22, 24, 25, 21, 26, 23, 39, 31, 19},
                            new Shape(1, 1, 3, 3));
            Assert.assertEquals(result, expected, "Sum Pooling operation failed");
        }
    }

    @Test
    public void testGlobalSumPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result = Pool.globalSumPool(original);
            NDArray expected = manager.create(new float[] {105}, new Shape(1, 1, 1, 1));
            Assert.assertEquals(result, expected, "Global Sum Pooling operation failed");
        }
    }

    @Test
    public void testAvgPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result =
                    Pool.avgPool(original, new Shape(2, 2), new Shape(2, 2), new Shape(0, 0));
            NDArray expected =
                    manager.create(new float[] {5.5f, 6.25f, 9.75f, 4.75f}, new Shape(1, 1, 2, 2));
            Assert.assertEquals(result, expected, "Avg Pooling operation failed");
        }
    }

    @Test
    public void testGlobalAvgPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result = Pool.globalAvgPool(original);
            NDArray expected = manager.create(new float[] {6.5625f}, new Shape(1, 1, 1, 1));
            Assert.assertEquals(result, expected, "Global Avg Pooling operation failed");
        }
    }

    @Test
    public void testLpPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result =
                    Pool.lpPool(original, new Shape(2, 2), new Shape(1, 1), new Shape(0, 0), 1);
            NDArray expected =
                    manager.create(
                            new float[] {22, 24, 25, 21, 26, 23, 39, 31, 19},
                            new Shape(1, 1, 3, 3));
            Assert.assertEquals(result, expected, "Sum Pooling operation failed");
        }
    }

    @Test
    public void testGlobalLpPool() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray result = Pool.globalLpPool(original, 1);
            NDArray expected = manager.create(new float[] {105}, new Shape(1, 1, 1, 1));
            Assert.assertEquals(result, expected, "Global Lp Pooling operation failed");
        }
    }
}
