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
package ai.djl.integration.tests.ndarray;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayElementComparisonOpTest {

    @Test
    public void testContentEquals() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f});
            NDArray array2 = manager.create(new float[] {1f, 2f});
            Assert.assertTrue(array1.contentEquals(array2));
            array1 = manager.ones(new Shape(2, 3));
            array2 = manager.ones(new Shape(1, 3));
            Assert.assertFalse(array1.contentEquals(array2));

            // test scalar
            array1 = manager.create(5f);
            array2 = manager.create(5f);
            Assert.assertTrue(array1.contentEquals(array2));
            array1 = manager.create(3);
            array2 = manager.create(4);
            Assert.assertFalse(array1.contentEquals(array2));

            // different data type
            array1 = manager.create(4f);
            array2 = manager.create(4);
            Assert.assertFalse(array1.contentEquals(array2));

            // test zero dim vs zero dim
            array1 = manager.create(new Shape(4, 0));
            array2 = manager.create(new Shape(4, 0));

            Assert.assertTrue(array1.contentEquals(array2));
            array1 = manager.create(new Shape(0, 0, 2));
            array2 = manager.create(new Shape(2, 0, 0));
            Assert.assertFalse(array1.contentEquals(array2));
        }
    }

    @Test
    public void testEqualsForScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray result = NDArrays.eq(array1, 2);
            NDArray expected = manager.create(new boolean[] {false, true, false, false});
            Assert.assertEquals(result, expected, "Incorrect comparison for equal NDArray");
            array1 = manager.ones(new Shape(4, 5, 2));
            result = NDArrays.eq(array1, 1);
            expected = manager.ones(new Shape(4, 5, 2)).toType(DataType.BOOLEAN, false);
            Assert.assertEquals(result, expected);

            array1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray array2 = manager.create(new float[] {1f, 3f, 3f, 4f});
            result = NDArrays.eq(array1, array2);
            expected = manager.create(new boolean[] {true, false, true, true});
            Assert.assertEquals(result, expected, "Incorrect comparison for unequal NDArray");

            // test scalar
            array1 = manager.create(4);
            result = NDArrays.eq(array1, 4);
            expected = manager.create(true);
            Assert.assertEquals(result, expected);

            // test zero-dim
            array1 = manager.create(new Shape(4, 3, 2, 1, 0));
            array2 = manager.create(new Shape(1, 0));
            result = NDArrays.eq(array1, array2);
            expected = manager.create(new Shape(4, 3, 2, 1, 0), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testEqualsForEqualNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray array2 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray result = NDArrays.eq(array1, array2);
            NDArray expected = manager.create(new boolean[] {true, true, true, true});
            Assert.assertEquals(result, expected, "Incorrect comparison for equal NDArray");
            array1 =
                    manager.create(
                            new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, new Shape(2, 5));
            array2 = manager.arange(10.0f).add(1).reshape(2, 5);
            result = NDArrays.eq(array1, array2);
            expected = manager.ones(new Shape(2, 5)).toType(DataType.BOOLEAN, false);
            Assert.assertEquals(result, expected);
            // test scalar
            array1 = manager.ones(new Shape(4)).mul(5);
            array2 = manager.create(5f);
            result = NDArrays.eq(array1, array2);
            expected = manager.create(new boolean[] {true, true, true, true});
            Assert.assertEquals(result, expected);
            // test zero-dim
            array1 = manager.create(new Shape(4, 3, 0));
            array2 = manager.create(new Shape(4, 3, 0));
            result = NDArrays.eq(array1, array2);
            expected = manager.create(new Shape(4, 3, 0), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testAllClose() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 0f, 2f, 2f, 4f});
            NDArray array2 = manager.create(new float[] {1.001f, 0.001f, 1.999f, 1.999f, 3.999f});
            Assert.assertTrue(array1.allClose(array2, 0, 0.01, false));
            Assert.assertTrue(array1.allClose(array2, 1, 0, false));
            Assert.assertTrue(NDArrays.allClose(array1, array2, 0, 0.01, false));
            Assert.assertTrue(NDArrays.allClose(array1, array2, 1, 0, false));

            array1 = manager.create(new double[] {1.0, Double.NaN});
            array2 = manager.create(new double[] {1.0, Double.NaN});
            Assert.assertFalse(array1.allClose(array2, 0, 0, false));
            Assert.assertTrue(array1.allClose(array2, 0, 0, true));
            Assert.assertFalse(NDArrays.allClose(array1, array2, 0, 0, false));
            Assert.assertTrue(NDArrays.allClose(array1, array2, 0, 0, true));

            array1 = manager.create(new double[] {Double.NaN, 2.0});
            array2 = manager.create(new double[] {2.0, Double.NaN});
            Assert.assertFalse(array1.allClose(array2, 0, 0, false));
            Assert.assertFalse(array1.allClose(array2, 0, 0, true));
            Assert.assertFalse(NDArrays.allClose(array1, array2, 0, 0, false));
            Assert.assertFalse(NDArrays.allClose(array1, array2, 0, 0, true));

            // test mult-dim
            array1 = manager.arange(10.0f, 20.0f).reshape(2, 5);
            array2 = array1.add(1e-5);
            Assert.assertTrue(array1.allClose(array2));
            Assert.assertTrue(NDArrays.allClose(array1, array2));

            // test scalar
            array1 = manager.create(5f);
            array2 = manager.create(5f + 1e-5);
            Assert.assertTrue(array1.allClose(array2));
            Assert.assertTrue(NDArrays.allClose(array1, array2));

            // test zero-dim
            array1 = manager.create(new Shape(0));
            array2 = manager.create(new Shape(0));
            Assert.assertTrue(array1.allClose(array2));
            Assert.assertTrue(array1.allClose(array2));
        }
    }

    @Test
    public void testGreaterThanScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 0f, 2f, 2f, 4f});
            NDArray result = NDArrays.gt(array, 2);
            NDArray expected = manager.create(new boolean[] {false, false, false, false, true});
            Assert.assertEquals(result, expected, "greater_scalar: Incorrect comparison");
            array =
                    manager.create(
                            new float[] {2f, 3f, -5f, 2f, 5f, 10f, 20123f, -355f},
                            new Shape(2, 2, 2));
            result = NDArrays.gt(array, 2);
            expected =
                    manager.create(
                            new boolean[] {false, true, false, false, true, true, true, false},
                            new Shape(2, 2, 2));
            Assert.assertEquals(result, expected);
            // test scalar
            array = manager.create(3f);
            result = NDArrays.gt(array, 3f);
            expected = manager.create(false);
            Assert.assertEquals(result, expected);
            // zero-dim
            array = manager.create(new Shape(2, 4, 0, 0, 1));
            result = NDArrays.gt(array, 0f);
            expected = manager.create(new Shape(2, 4, 0, 0, 1), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testGreaterThanNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f, 4f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 2f, 5f, 4f, 5f});
            NDArray result = NDArrays.gt(array1, array2);
            NDArray expected =
                    manager.create(new boolean[] {false, true, false, false, true, false});
            Assert.assertEquals(result, expected, "greater: Incorrect comparison");
            array1 = manager.create(new float[] {0f, 3f, 5f, 7f, 10f, 3f, 2f, 2f}, new Shape(2, 4));
            array2 =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 10f, 3f, -234f, 66f}, new Shape(2, 4));
            result = NDArrays.gt(array1, array2);
            expected =
                    manager.create(
                            new boolean[] {true, false, true, false, false, false, true, false},
                            new Shape(2, 4));
            Assert.assertEquals(result, expected);
            // test scalar with scalar
            array1 = manager.create(4f);
            array2 = manager.create(4f);
            result = NDArrays.gt(array1, array2);
            expected = manager.create(false);
            Assert.assertEquals(result, expected);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.gt(array1, array2);
            expected = manager.create(new boolean[] {false, false, false, true}, new Shape(2, 2));
            Assert.assertEquals(result, expected);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.gt(array1, array2);
            expected = manager.create(new Shape(0, 0, 0), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testGreaterThanOrEqualToScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 2f, 4f});
            NDArray result = NDArrays.gte(array, 2);
            NDArray expected = manager.create(new boolean[] {false, true, true, true});
            Assert.assertEquals(result, expected, "greater_equals_scalar: Incorrect comparison");
            array = manager.create(new float[] {3f, 2f, 2f, 4f, 5f, 3f}, new Shape(3, 2));
            result = NDArrays.gte(array, 3f);
            expected =
                    manager.create(
                            new boolean[] {true, false, false, true, true, true}, new Shape(3, 2));
            Assert.assertEquals(result, expected, "greater_equals_scalar: Incorrect comparison");
            // test scalar
            array = manager.create(4f);
            result = NDArrays.gt(array, 4);
            expected = manager.create(false);
            Assert.assertEquals(result, expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0, 1));
            result = NDArrays.gt(array, 2f);
            expected = manager.create(new Shape(0, 0, 1), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testGreaterThanOrEqualToNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f, 4f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 2f, 5f, 4f, 5f});
            NDArray result = NDArrays.gte(array1, array2);
            NDArray expected =
                    manager.create(new boolean[] {false, true, true, false, true, false});
            Assert.assertEquals(result, expected, "greater_equal: Incorrect comparison");
            array1 =
                    manager.create(
                            new float[] {3f, 2.19f, 3.1f, -3.2f, -4, -2, -1.1f, -2.3f},
                            new Shape(2, 1, 2, 1, 2));
            array2 = manager.ones(new Shape(2, 1, 2, 1, 2)).mul(2.2f);
            result = NDArrays.gte(array1, array2);
            expected =
                    manager.create(
                            new boolean[] {true, false, true, false, false, false, false, false},
                            new Shape(2, 1, 2, 1, 2));
            Assert.assertEquals(result, expected);
            // test scalar with scalar
            array1 = manager.create(4f);
            array2 = manager.create(4f);
            result = NDArrays.gte(array1, array2);
            expected = manager.create(true);
            Assert.assertEquals(result, expected);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.gte(array1, array2);
            expected = manager.create(new boolean[] {true, true, true, true}, new Shape(2, 2));
            Assert.assertEquals(result, expected);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.gt(array1, array2);
            expected = manager.create(new Shape(0, 0, 0), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testLesserThanScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 2f, 4f, 5f});
            NDArray result = NDArrays.lt(array, 2);
            NDArray expected = manager.create(new boolean[] {true, false, false, false, false});
            Assert.assertEquals(result, expected, "lesser_scalar: Incorrect comparison");
            array =
                    manager.create(
                            new float[] {2.2322f, 2.3222f, 2.3333f, 2.2222f}, new Shape(2, 2));
            result = NDArrays.lt(array, 2.3322f);
            expected = manager.create(new boolean[] {true, true, false, true}, new Shape(2, 2));
            Assert.assertEquals(result, expected);
            // test scalar
            array = manager.create(3.9999f);
            result = NDArrays.lt(array, 4);
            expected = manager.create(true);
            Assert.assertEquals(result, expected);
            // test zero-dim
            array = manager.create(new Shape(2, 4, 3, 5, 1, 0));
            result = NDArrays.lt(array, 2f);
            expected = manager.create(new Shape(2, 4, 3, 5, 1, 0), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testLesserThanNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 1f, 5f, 4f});
            NDArray result = NDArrays.lt(array1, array2);
            NDArray expected = manager.create(new boolean[] {true, false, false, true, false});
            Assert.assertEquals(result, expected, "lesser_scalar: Incorrect comparison");
            array1 =
                    manager.create(
                            new float[] {1.1f, 2f, 1.534f, 2.001f, 2.000001f, 2.22f},
                            new Shape(2, 1, 3));
            array2 =
                    manager.create(
                            new float[] {1.011f, 2.01f, 1.5342f, 2.000001f, 2.01f, 2.3f},
                            new Shape(2, 1, 3));
            result = NDArrays.lt(array1, array2);
            expected =
                    manager.create(
                            new boolean[] {false, true, true, false, true, true},
                            new Shape(2, 1, 3));
            Assert.assertEquals(result, expected);
            // test scalar with scalar
            array1 = manager.create(4.1f);
            array2 = manager.create(4.1f);
            result = NDArrays.lt(array1, array2);
            expected = manager.create(false);
            Assert.assertEquals(result, expected);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.arange(10.0f).reshape(new Shape(2, 5, 1));
            result = NDArrays.lt(array1, array2);
            expected =
                    manager.create(
                            new boolean[] {
                                false, false, false, false, true, true, true, true, true, true
                            },
                            new Shape(2, 5, 1));
            Assert.assertEquals(result, expected);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(2, 0, 1));
            array2 = manager.create(new Shape(1, 0, 1));
            result = NDArrays.lt(array1, array2);
            expected = manager.create(new Shape(2, 0, 1), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testLesserThanOrEqualToScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10.0f);
            NDArray result = NDArrays.lte(array, 4);
            NDArray expected =
                    manager.create(
                            new boolean[] {
                                true, true, true, true, true, false, false, false, false, false
                            });
            Assert.assertEquals(result, expected);
            array = manager.create(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            result = NDArrays.lte(array, 0.2f);
            expected = manager.create(new boolean[] {true, true, false, false}, new Shape(2, 2, 1));
            Assert.assertEquals(result, expected);
            // test scalar
            array = manager.create(3.9999f);
            result = NDArrays.lt(array, 4);
            expected = manager.create(true);
            Assert.assertEquals(result, expected);
            // test zero-dim
            array = manager.create(new Shape(2, 0, 3, 0, 1, 0));
            result = NDArrays.lt(array, 2f);
            expected = manager.create(new Shape(2, 0, 3, 0, 1, 0), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testLesserThanOrEqualToNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.arange(10.0f);
            NDArray array2 = manager.ones(new Shape(10)).mul(5);
            NDArray result = NDArrays.lte(array1, array2);
            NDArray expected =
                    manager.create(
                            new boolean[] {
                                true, true, true, true, true, true, false, false, false, false
                            });
            Assert.assertEquals(result, expected);
            array1 = manager.create(new float[] {2f, 3f, 4f, 5f}, new Shape(2, 2));
            array2 = manager.arange(4.0f).add(1).reshape(1, 2, 2);
            result = NDArrays.lte(array1, array2);
            expected =
                    manager.create(new boolean[] {false, false, false, false}, new Shape(1, 2, 2));
            Assert.assertEquals(result, expected);
            // test scalar with scalar
            array1 = manager.create(0f);
            array2 = manager.create(0f);
            result = NDArrays.lte(array1, array2);
            expected = manager.create(true);
            Assert.assertEquals(result, expected);
            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = NDArrays.lte(array1, array2);
            expected = manager.create(new boolean[] {true, true, true, false}, new Shape(2, 2));
            Assert.assertEquals(result, expected);
            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = NDArrays.lte(array1, array2);
            expected = manager.create(new Shape(0, 0, 0), DataType.BOOLEAN);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testMaximumScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10.0f);
            NDArray result = array.maximum(4.0f);
            NDArray expected = manager.create(new float[] {4f, 4f, 4f, 4f, 4f, 5f, 6f, 7f, 8f, 9f});
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array, 4f);
            Assert.assertEquals(result, expected);

            array = manager.create(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            result = array.maximum(0.2f);
            expected = manager.create(new float[] {0.2f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array, 0.2f);
            Assert.assertEquals(result, expected);

            // test scalar
            array = manager.create(3.9999f);
            result = array.maximum(4.0f);
            expected = manager.create(4f);
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array, 4f);
            Assert.assertEquals(result, expected);

            // test zero-dim
            array = manager.create(new Shape(2, 0, 3, 0, 1, 0));
            result = array.maximum(2f);
            expected = manager.create(new Shape(2, 0, 3, 0, 1, 0));
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array, 2f);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testMaximumNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f});
            NDArray array2 = manager.create(new float[] {5f, 4f, 3f, 2f, 1f});
            NDArray result = array1.maximum(array2);
            NDArray expected = manager.create(new float[] {5f, 4f, 3f, 4f, 5f});
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array1, array2);
            Assert.assertEquals(result, expected);

            array1 = manager.arange(10.0f).reshape(new Shape(2, 5));
            array2 = manager.create(new float[] {4f, 5f}, new Shape(2, 1));
            result = array1.maximum(array2);
            expected =
                    manager.create(
                            new float[] {4f, 4f, 4f, 4f, 4f, 5f, 6f, 7f, 8f, 9f}, new Shape(2, 5));
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array1, array2);
            Assert.assertEquals(result, expected);

            // test scalar with scalar
            array1 = manager.create(0f);
            array2 = manager.create(1f);
            result = array1.maximum(array2);
            expected = manager.create(1f);
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array1, array2);
            Assert.assertEquals(result, expected);

            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = array1.maximum(array2);
            expected = manager.ones(new Shape(2, 2)).mul(3);
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array1, array2);
            Assert.assertEquals(result, expected);

            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = array1.maximum(array2);
            expected = manager.create(new Shape(0, 0, 0));
            Assert.assertEquals(result, expected);
            result = NDArrays.maximum(array1, array2);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testMinimumScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10.0f);
            NDArray result = array.minimum(4f);
            NDArray expected = manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 4f, 4f, 4f, 4f, 4f});
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array, 4f);
            Assert.assertEquals(result, expected);

            array = manager.create(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2, 1));
            result = array.minimum(0.2f);
            expected = manager.create(new float[] {0.1f, 0.2f, 0.2f, 0.2f}, new Shape(2, 2, 1));
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array, 0.2f);
            Assert.assertEquals(result, expected);

            // test scalar
            array = manager.create(3.9999f);
            result = array.minimum(4f);
            expected = manager.create(3.9999f);
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array, 4f);
            Assert.assertEquals(result, expected);

            // test zero-dim
            array = manager.create(new Shape(1, 0));
            result = array.minimum(2f);
            expected = manager.create(new Shape(1, 0));
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array, 2f);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testMinimumNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f});
            NDArray array2 = manager.create(new float[] {5f, 4f, 3f, 2f, 1f});
            NDArray result = array1.minimum(array2);
            NDArray expected = manager.create(new float[] {1f, 2f, 3f, 2f, 1f});
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array1, array2);
            Assert.assertEquals(result, expected);

            array1 = manager.arange(10.0f).reshape(new Shape(2, 5));
            array2 = manager.create(new float[] {4f, 5f}, new Shape(2, 1));
            result = array1.minimum(array2);
            expected =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 5f, 5f, 5f, 5f}, new Shape(2, 5));
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array1, array2);
            Assert.assertEquals(result, expected);

            // test scalar with scalar
            array1 = manager.create(0f);
            array2 = manager.create(1f);
            result = array1.minimum(array2);
            expected = manager.create(0f);
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array1, array2);
            Assert.assertEquals(result, expected);

            // test NDArray with scalar
            array1 = manager.create(3f);
            array2 = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            result = array1.minimum(array2);
            expected = manager.create(new float[] {3f, 3f, 3f, 2f}, new Shape(2, 2));
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array1, array2);
            Assert.assertEquals(result, expected);

            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(0, 0, 1));
            array2 = manager.create(new Shape(1, 0, 0));
            result = array1.minimum(array2);
            expected = manager.create(new Shape(0, 0, 0));
            Assert.assertEquals(result, expected);
            result = NDArrays.minimum(array1, array2);
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testWhere() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 2f, 4f, 5f, 4f});
            NDArray array2 = manager.create(new float[] {2f, 1f, 3f, 5f, 4f, 5f});
            NDArray condition =
                    manager.create(new boolean[] {true, true, false, true, false, false});
            NDArray result = NDArrays.where(condition, array1, array2);
            NDArray expected = manager.create(new float[] {1f, 2f, 3f, 4f, 4f, 5f});
            Assert.assertEquals(result, expected, "where: Incorrect comparison");

            array1 = manager.create(new float[] {0f, 3f, 5f, 7f, 10f, 3f, 2f, 2f}, new Shape(2, 4));
            array2 =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 10f, 3f, -234f, 66f}, new Shape(2, 4));
            condition =
                    manager.create(
                            new boolean[] {false, true, false, true, true, true, false, true},
                            new Shape(2, 4));
            expected =
                    manager.create(
                            new float[] {-2f, 3f, 2f, 7f, 10f, 3f, -234f, 2f}, new Shape(2, 4));
            result = NDArrays.where(condition, array1, array2);
            Assert.assertEquals(result, expected, "where: Incorrect comparison");

            // test cond broadcasting
            array1 =
                    manager.create(
                            new float[] {0f, 3f, 5f, 9f, 11f, 12f, -2f, -4f}, new Shape(2, 4));
            array2 =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 10f, 3f, -234f, 66f}, new Shape(2, 4));
            condition = manager.create(new boolean[] {false, true});
            expected =
                    manager.create(
                            new float[] {-2f, 43f, 2f, 7f, 11f, 12f, -2f, -4f}, new Shape(2, 4));
            result = NDArrays.where(condition, array1, array2);
            Assert.assertEquals(result, expected, "where: Incorrect comparison");
            // test x, y broadcasting
            array1 = manager.create(new float[] {0f, 1f, 2f}).reshape(3, 1);
            array2 = manager.create(new float[] {3f, 4f, 5f, 6f}).reshape(1, 4);
            condition =
                    manager.create(
                            new boolean[] {
                                false, true, true, true, false, false, true, true, false, false,
                                false, true
                            },
                            new Shape(3, 4));
            result = NDArrays.where(condition, array1, array2);
            expected =
                    manager.create(
                            new float[] {3f, 0f, 0f, 0f, 3f, 4f, 1f, 1f, 3f, 4f, 5f, 2f},
                            new Shape(3, 4));
            Assert.assertEquals(result, expected, "where: Incorrect comparison");

            // test where examples in numpy website
            array1 = manager.arange(10.0f);
            result = NDArrays.where(array1.lt(5), array1, array1.mul(10));
            expected = manager.create(new float[] {0, 1, 2, 3, 4, 50, 60, 70, 80, 90});
            Assert.assertEquals(result, expected);

            array1 =
                    manager.create(
                            new float[] {0f, 1f, 2f, 0f, 2f, 4f, 0f, 3f, 6f}, new Shape(3, 3));
            array2 = manager.create(new float[] {-1f});
            result = NDArrays.where(array1.lt(4), array1, array2);
            expected =
                    manager.create(
                            new float[] {0f, 1f, 2f, 0f, 2f, -1f, 0f, 3f, -1f}, new Shape(3, 3));
            Assert.assertEquals(result, expected);

            // test scalar with scalar
            array1 = manager.create(4f);
            array2 = manager.create(6f);
            condition = manager.create(false);
            result = NDArrays.where(condition, array1, array2);
            expected = manager.create(6f);
            Assert.assertEquals(result, expected, "where: Incorrect comparison");

            // test zero-dim
            array1 = manager.create(new Shape(1, 0, 0));
            array2 = manager.create(new Shape(1, 0, 0));
            condition = manager.create(new Shape(1, 0, 0));
            result = NDArrays.where(condition, array1, array2);
            expected = manager.create(new Shape(1, 0, 0));
            Assert.assertEquals(result, expected, "where: Incorrect comparison");
        }
    }
}
