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

import org.testng.annotations.Test;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayElementComparisonOpTest {

    @Test
    public void testContentEquals() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f});
            NDArray array2 = manager.create(new float[] {1f, 2f});
            Assertions.assertTrue(array1.contentEquals(array2) && NDArrays.equals(array1, array2));
            array1 = manager.ones(new Shape(2, 3));
            array2 = manager.ones(new Shape(1, 3));
            Assertions.assertFalse(array1.contentEquals(array2) && NDArrays.equals(array1, array2));

            // test scalar
            array1 = manager.create(5f);
            array2 = manager.create(5f);
            Assertions.assertTrue(array1.contentEquals(array2) && NDArrays.equals(array1, array2));
            array1 = manager.create(3);
            array2 = manager.create(4);
            Assertions.assertFalse(array1.contentEquals(array2) && NDArrays.equals(array1, array2));

            // different data type
            array1 = manager.create(4f);
            array2 = manager.create(4);
            Assertions.assertFalse(array1.contentEquals(array2) || NDArrays.equals(array1, array2));

            // test zero dim vs zero dim
            array1 = manager.create(new Shape(4, 0));
            array2 = manager.create(new Shape(4, 0));

            Assertions.assertTrue(array1.contentEquals(array2) && NDArrays.equals(array1, array2));
            array1 = manager.create(new Shape(0, 0, 2));
            array2 = manager.create(new Shape(2, 0, 0));
            Assertions.assertFalse(array1.contentEquals(array2) && NDArrays.equals(array1, array2));
        }
    }

    @Test
    public void testEqualsForScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray result = NDArrays.eq(array1, 2);
            NDArray actual = manager.create(new float[] {0f, 1f, 0f, 0f});
            Assertions.assertEquals(actual, result, "Incorrect comparison for equal NDArray");
            array1 = manager.ones(new Shape(4, 5, 2));
            result = NDArrays.eq(array1, 1);
            actual = manager.ones(new Shape(4, 5, 2));
            Assertions.assertEquals(actual, result);

            array1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray array2 = manager.create(new float[] {1f, 3f, 3f, 4f});
            result = NDArrays.eq(array1, array2);
            actual = manager.create(new float[] {1f, 0f, 1f, 1f});
            Assertions.assertEquals(actual, result, "Incorrect comparison for unequal NDArray");

            // test scalar
            array1 = manager.create(4);
            result = NDArrays.eq(array1, 4);
            actual = manager.create(1);
            Assertions.assertEquals(actual, result);

            // test zero-dim
            array1 = manager.create(new Shape(4, 3, 2, 1, 0));
            array2 = manager.create(new Shape(1, 0));
            result = NDArrays.eq(array1, array2);
            Assertions.assertEquals(manager.create(new Shape(4, 3, 2, 1, 0)), result);
        }
    }

    @Test
    public void testEqualsForEqualNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray array2 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray result = NDArrays.eq(array1, array2);
            NDArray actual = manager.ones(new Shape(4));
            Assertions.assertEquals(actual, result, "Incorrect comparison for equal NDArray");
            array1 =
                    manager.create(
                            new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, new Shape(2, 5));
            array2 = manager.arange(10).add(1).reshape(2, 5);
            result = NDArrays.eq(array1, array2);
            actual = manager.ones(new Shape(2, 5));
            Assertions.assertEquals(actual, result);
            // test scalar
            array1 = manager.ones(new Shape(4)).mul(5);
            array2 = manager.create(5f);
            result = NDArrays.eq(array1, array2);
            actual = manager.ones(new Shape(4));
            Assertions.assertEquals(actual, result);
            // test zero-dim
            array1 = manager.create(new Shape(4, 3, 0));
            array2 = manager.create(new Shape(4, 3, 0));
            result = NDArrays.eq(array1, array2);
            actual = manager.create(new Shape(4, 3, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testGreaterThanScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 0f, 2f, 2f, 4f});
            NDArray result = NDArrays.gt(array, 2);
            NDArray actual = manager.create(new float[] {0f, 0f, 0f, 0f, 1f});
            Assertions.assertEquals(actual, result, "greater_scalar: Incorrect comparison");
            array =
                    manager.create(
                            new float[] {2f, 3f, -5f, 2f, 5f, 10f, 20123f, -355f},
                            new Shape(2, 2, 2));
            result = NDArrays.gt(array, 2);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 0f, 0f, 1f, 1f, 1f, 0f}, new Shape(2, 2, 2));
            Assertions.assertEquals(actual, result);
            // test scalar
            array = manager.create(3f);
            result = NDArrays.gt(array, 3f);
            actual = manager.create(0f);
            Assertions.assertEquals(actual, result);
            // zero-dim
            array = manager.create(new Shape(2, 4, 0, 0, 1));
            result = NDArrays.gt(array, 0f);
            Assertions.assertEquals(array, result);
        }
    }

    @Test
    public void testGreaterThanNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f, 4f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 2f, 5f, 4f, 5f});
            NDArray result = NDArrays.gt(array1, array2);
            NDArray actual = manager.create(new float[] {0f, 1f, 0f, 0f, 1f, 0f});
            Assertions.assertEquals(actual, result, "greater: Incorrect comparison");
            array1 = manager.create(new float[] {0f, 3f, 5f, 7f, 10f, 3f, 2f, 2f}, new Shape(2, 4));
            array2 =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 10f, 3f, -234f, 66f}, new Shape(2, 4));
            result = NDArrays.gt(array1, array2);
            actual = manager.create(new float[] {1f, 0f, 1f, 0f, 0f, 0f, 1f, 0f}, new Shape(2, 4));
            Assertions.assertEquals(actual, result);
            // test scalar with scalar
            array1 = manager.create(4f);
            array2 = manager.create(4f);
            result = NDArrays.gt(array1, array2);
            actual = manager.create(0f);
            Assertions.assertEquals(actual, result);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.gt(array1, array2);
            actual = manager.create(new float[] {0f, 0f, 0f, 1f}, new Shape(2, 2));
            Assertions.assertEquals(actual, result);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.gt(array1, array2);
            actual = manager.create(new Shape(0, 0, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testWhere() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f, 4f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 3f, 5f, 4f, 5f});
            NDArray condition = manager.create(new float[] {1, 1, 0, 1, 0, 0});
            NDArray result = NDArrays.where(condition, array1, array2);
            NDArray actual = manager.create(new float[] {1f, 2f, 3f, 4f, 4f, 5f});
            Assertions.assertEquals(result, actual, "where: Incorrect comparison");

            array1 = manager.create(new float[] {0f, 3f, 5f, 7f, 10f, 3f, 2f, 2f}, new Shape(2, 4));
            array2 =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 10f, 3f, -234f, 66f}, new Shape(2, 4));
            condition =
                    manager.create(new float[] {0f, 1f, 0f, 1f, 1f, 1f, 0f, 1f}, new Shape(2, 4));
            actual =
                    manager.create(
                            new float[] {-2f, 3f, 2f, 7f, 10f, 3f, -234f, 2f}, new Shape(2, 4));
            result = NDArrays.where(condition, array1, array2);
            Assertions.assertEquals(result, actual, "where: Incorrect comparison");

            // test with broadcasting
            array1 =
                    manager.create(
                            new float[] {0f, 3f, 5f, 9f, 11f, 12f, -2f, -4f}, new Shape(2, 4));
            array2 =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 10f, 3f, -234f, 66f}, new Shape(2, 4));
            condition = manager.create(new float[] {0f, 1f}, new Shape(2));
            actual =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 11f, 12f, -2f, -4f}, new Shape(2, 4));
            result = NDArrays.where(condition, array1, array2);
            Assertions.assertEquals(result, actual, "where: Incorrect comparison");

            // test scalar with scalar
            array1 = manager.create(4f);
            array2 = manager.create(6f);
            condition = manager.create(0f);
            result = NDArrays.where(condition, array1, array2);
            actual = manager.create(6f);
            Assertions.assertEquals(result, actual, "where: Incorrect comparison");

            // test zero-dim
            array1 = manager.create(new Shape(1, 0, 0));
            array2 = manager.create(new Shape(1, 0, 0));
            condition = manager.create(new Shape(1, 0, 0));
            result = NDArrays.where(condition, array1, array2);
            actual = manager.create(new Shape(1, 0, 0));
            Assertions.assertEquals(result, actual, "where: Incorrect comparison");
        }
    }

    @Test
    public void testGreaterThanOrEqualToScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 2f, 4f});
            NDArray result = NDArrays.gte(array, 2);
            NDArray actual = manager.create(new float[] {0f, 1f, 1f, 1f});
            Assertions.assertEquals(actual, result, "greater_equals_scalar: Incorrect comparison");
            array = manager.create(new float[] {3f, 2f, 2f, 4f, 5f, 3f}, new Shape(3, 2));
            result = NDArrays.gte(array, 3f);
            actual = manager.create(new float[] {1f, 0f, 0f, 1f, 1f, 1f}, new Shape(3, 2));
            Assertions.assertEquals(actual, result, "greater_equals_scalar: Incorrect comparison");
            // test scalar
            array = manager.create(4f);
            result = NDArrays.gt(array, 4);
            actual = manager.create(0f);
            Assertions.assertEquals(actual, result);
            // test zero-dim
            array = manager.create(new Shape(0, 0, 1));
            result = NDArrays.gt(array, 2f);
            actual = manager.create(new Shape(0, 0, 1));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testGreaterThanOrEqualToNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f, 4f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 2f, 5f, 4f, 5f});
            NDArray result = NDArrays.gte(array1, array2);
            NDArray actual = manager.create(new float[] {0f, 1f, 1f, 0f, 1f, 0f});
            Assertions.assertEquals(actual, result, "greater_equal: Incorrect comparison");
            array1 =
                    manager.create(
                            new float[] {3f, 2.19f, 3.1f, -3.2f, -4, -2, -1.1f, -2.3f},
                            new Shape(2, 1, 2, 1, 2));
            array2 = manager.ones(new Shape(2, 1, 2, 1, 2)).mul(2.2f);
            result = NDArrays.gte(array1, array2);
            actual =
                    manager.create(
                            new float[] {1f, 0f, 1f, 0f, 0f, 0f, 0f, 0f}, new Shape(2, 1, 2, 1, 2));
            Assertions.assertEquals(actual, result);
            // test scalar with scalar
            array1 = manager.create(4f);
            array2 = manager.create(4f);
            result = NDArrays.gte(array1, array2);
            actual = manager.create(1f);
            Assertions.assertEquals(actual, result);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.gte(array1, array2);
            actual = manager.create(new float[] {1f, 1f, 1f, 1f}, new Shape(2, 2));
            Assertions.assertEquals(actual, result);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.gt(array1, array2);
            actual = manager.create(new Shape(0, 0, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testLesserThanScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 2f, 4f, 5f});
            NDArray result = NDArrays.lt(array, 2);
            NDArray actual = manager.create(new float[] {1f, 0f, 0f, 0f, 0f});
            Assertions.assertEquals(actual, result, "lesser_scalar: Incorrect comparison");
            array =
                    manager.create(
                            new float[] {2.2322f, 2.3222f, 2.3333f, 2.2222f}, new Shape(2, 2));
            result = NDArrays.lt(array, 2.3322f);
            actual = manager.create(new float[] {1f, 1f, 0f, 1f}, new Shape(2, 2));
            Assertions.assertEquals(actual, result);
            // test scalar
            array = manager.create(3.9999f);
            result = NDArrays.lt(array, 4);
            actual = manager.create(1f);
            Assertions.assertEquals(actual, result);
            // test zero-dim
            array = manager.create(new Shape(2, 4, 3, 5, 1, 0));
            result = NDArrays.lt(array, 2f);
            actual = manager.create(new Shape(2, 4, 3, 5, 1, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testLesserThanNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 1f, 5f, 4f});
            NDArray result = NDArrays.lt(array1, array2);
            NDArray actual = manager.create(new float[] {1f, 0f, 0f, 1f, 0f});
            Assertions.assertEquals(actual, result, "lesser_scalar: Incorrect comparison");
            array1 =
                    manager.create(
                            new float[] {1.1f, 2f, 1.534f, 2.001f, 2.000001f, 2.22f},
                            new Shape(2, 1, 3));
            array2 =
                    manager.create(
                            new float[] {1.011f, 2.01f, 1.5342f, 2.000001f, 2.01f, 2.3f},
                            new Shape(2, 1, 3));
            result = NDArrays.lt(array1, array2);
            actual = manager.create(new float[] {0f, 1f, 1f, 0f, 1f, 1f}, new Shape(2, 1, 3));
            Assertions.assertEquals(actual, result);
            // test scalar with scalar
            array1 = manager.create(4.1f);
            array2 = manager.create(4.1f);
            result = NDArrays.lt(array1, array2);
            actual = manager.create(0f);
            Assertions.assertEquals(actual, result);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.arange(10).reshape(new Shape(2, 5, 1));
            result = NDArrays.lt(array1, array2);
            actual =
                    manager.create(
                            new float[] {0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f, 1f, 1f},
                            new Shape(2, 5, 1));
            Assertions.assertEquals(actual, result);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(2, 0, 1));
            array2 = manager.create(new Shape(1, 0, 1));
            result = NDArrays.lt(array1, array2);
            actual = manager.create(new Shape(2, 0, 1));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testLesserThanOrEqualToScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10);
            NDArray result = NDArrays.lte(array, 4);
            NDArray actual = manager.create(new float[] {1f, 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f, 0f});
            Assertions.assertEquals(actual, result);
            array = manager.create(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            result = NDArrays.lte(array, 0.2f);
            actual = manager.create(new float[] {1f, 1f, 0f, 0f}, new Shape(2, 2, 1));
            Assertions.assertEquals(actual, result);
            // test scalar
            array = manager.create(3.9999f);
            result = NDArrays.lt(array, 4);
            actual = manager.create(1f);
            Assertions.assertEquals(actual, result);
            // test zero-dim
            array = manager.create(new Shape(2, 0, 3, 0, 1, 0));
            result = NDArrays.lt(array, 2f);
            actual = manager.create(new Shape(2, 0, 3, 0, 1, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testLesserThanOrEqualToNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.arange(10);
            NDArray array2 = manager.ones(new Shape(10)).mul(5);
            NDArray result = NDArrays.lte(array1, array2);
            NDArray actual = manager.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f});
            Assertions.assertEquals(actual, result);
            array1 = manager.create(new float[] {2f, 3f, 4f, 5f}, new Shape(2, 2));
            array2 = manager.arange(4).add(1).reshape(1, 2, 2);
            result = NDArrays.lte(array1, array2);
            actual = manager.create(new float[] {0f, 0f, 0f, 0f}, new Shape(1, 2, 2));
            Assertions.assertEquals(actual, result);
            // test scalar with scalar
            array1 = manager.create(0f);
            array2 = manager.create(0f);
            result = NDArrays.lte(array1, array2);
            actual = manager.create(1f);
            Assertions.assertEquals(actual, result);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.lte(array1, array2);
            actual = manager.create(new float[] {1f, 1f, 1f, 0f}, new Shape(2, 2));
            Assertions.assertEquals(actual, result);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.lte(array1, array2);
            actual = manager.create(new Shape(0, 0, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testMaxScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10);
            NDArray result = NDArrays.max(array, 4);
            NDArray actual = manager.create(new float[] {4f, 4f, 4f, 4f, 4f, 5f, 6f, 7f, 8f, 9f});
            Assertions.assertEquals(actual, result);
            array = manager.create(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            result = NDArrays.max(array, 0.2f);
            actual = manager.create(new float[] {0.2f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            Assertions.assertEquals(actual, result);
            // test scalar
            array = manager.create(3.9999f);
            result = NDArrays.max(array, 4);
            actual = manager.create(4f);
            Assertions.assertEquals(actual, result);
            // test zero-dim
            array = manager.create(new Shape(2, 0, 3, 0, 1, 0));
            result = NDArrays.max(array, 2f);
            actual = manager.create(new Shape(2, 0, 3, 0, 1, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testMaxNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f});
            NDArray array2 = manager.create(new float[] {5f, 4f, 3f, 2f, 1f});
            NDArray result = NDArrays.max(array1, array2);
            NDArray actual = manager.create(new float[] {5f, 4f, 3f, 4f, 5f});
            Assertions.assertEquals(actual, result);
            array1 = manager.arange(10).reshape(new Shape(2, 5));
            array2 = manager.create(new float[] {4f, 5f}, new Shape(2, 1));
            result = NDArrays.max(array1, array2);
            actual =
                    manager.create(
                            new float[] {4f, 4f, 4f, 4f, 4f, 5f, 6f, 7f, 8f, 9f}, new Shape(2, 5));
            Assertions.assertEquals(actual, result);
            // test scalar with scalar
            array1 = manager.create(0f);
            array2 = manager.create(1f);
            result = NDArrays.max(array1, array2);
            actual = manager.create(1f);
            Assertions.assertEquals(actual, result);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.max(array1, array2);
            actual = manager.ones(new Shape(2, 2)).mul(3);
            Assertions.assertEquals(actual, result);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.lte(array1, array2);
            actual = manager.create(new Shape(0, 0, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testMinScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10);
            NDArray result = NDArrays.min(array, 4);
            NDArray actual = manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 4f, 4f, 4f, 4f, 4f});
            Assertions.assertEquals(actual, result);
            array = manager.create(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            result = NDArrays.min(array, 0.2f);
            actual = manager.create(new float[] {0.1f, 0.2f, 0.2f, 0.2f}, new Shape(2, 2, 1));
            Assertions.assertEquals(actual, result);
            // test scalar
            array = manager.create(3.9999f);
            result = NDArrays.min(array, 4);
            actual = manager.create(3.9999f);
            Assertions.assertEquals(actual, result);
            // test zero-dim
            array = manager.create(new Shape(1, 0));
            result = NDArrays.min(array, 2f);
            actual = manager.create(new Shape(1, 0));
            Assertions.assertEquals(actual, result);
        }
    }

    @Test
    public void testMinNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f});
            NDArray array2 = manager.create(new float[] {5f, 4f, 3f, 2f, 1f});
            NDArray result = NDArrays.min(array1, array2);
            NDArray actual = manager.create(new float[] {1f, 2f, 3f, 2f, 1f});
            Assertions.assertEquals(actual, result);
            array1 = manager.arange(10).reshape(new Shape(2, 5));
            array2 = manager.create(new float[] {4f, 5f}, new Shape(2, 1));
            result = NDArrays.min(array1, array2);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 5f, 5f, 5f, 5f}, new Shape(2, 5));
            Assertions.assertEquals(actual, result);
            // test scalar with scalar
            array1 = manager.create(0f);
            array2 = manager.create(1f);
            result = NDArrays.min(array1, array2);
            actual = manager.create(0f);
            Assertions.assertEquals(actual, result);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.min(array1, array2);
            actual = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            Assertions.assertEquals(actual, result);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.min(array1, array2);
            actual = manager.create(new Shape(0, 0, 0));
            Assertions.assertEquals(actual, result);
        }
    }
}
