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

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayOtherOpTest {

    @Test
    public void testGet() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            Assert.assertEquals(original, original.get(new NDIndex()));

            NDArray getAt = original.get(0);
            NDArray actual = manager.create(new float[] {1f, 2f});
            Assert.assertEquals(actual, getAt);

            Assert.assertEquals(actual, original.get("0,:"));
            Assert.assertEquals(actual, original.get("0,*"));

            NDArray getSlice = original.get("1:");
            actual = manager.create(new float[] {3f, 4f}, new Shape(1, 2));
            Assert.assertEquals(actual, getSlice);

            NDArray getStepSlice = original.get("1::2");
            Assert.assertEquals(actual, getStepSlice);

            // get from boolean array
            original = manager.arange(10).reshape(2, 5);
            NDArray bool = manager.create(new boolean[] {true, false});
            actual = manager.arange(5).reshape(1, 5);
            Assert.assertEquals(actual, original.get(bool));
        }
    }

    @Test
    public void testSetArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray actual = manager.create(new float[] {9, 10, 3, 4}, new Shape(2, 2));
            NDArray value = manager.create(new float[] {9, 10});
            original.set(new NDIndex(0), value);
            Assert.assertEquals(actual, original);
        }
    }

    @Test
    public void testSetArrayBroadcast() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2, 1));
            NDArray actual = manager.create(new float[] {9, 9, 3, 4}, new Shape(2, 2, 1));
            NDArray value = manager.create(new float[] {9});
            original.set(new NDIndex(0), value);
            Assert.assertEquals(actual, original);
        }
    }

    @Test
    public void testSetNumber() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray actual = manager.create(new float[] {9, 9, 3, 4}, new Shape(2, 2));
            original.set(new NDIndex(0), 9);
            Assert.assertEquals(actual, original);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testSetScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            original.setScalar(new NDIndex(0, 0), 0);
            original.setScalar(new NDIndex(0), 1);
        }
    }

    @Test
    public void testCopyTo() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray array2 = manager.create(new Shape(4));
            array1.copyTo(array2);
            Assert.assertEquals(array1, array2, "CopyTo NDArray failed");
            // test multi-dim
            array1 = manager.arange(100).reshape(2, 5, 5, 2);
            array2 = manager.create(new Shape(2, 5, 5, 2));
            array1.copyTo(array2);
            Assert.assertEquals(array1, array2, "CopyTo NDArray failed");
            // test scalar
            array1 = manager.create(5f);
            array2 = manager.create(new Shape());
            array1.copyTo(array2);
            Assert.assertEquals(array1, array2, "CopyTo NDArray failed");
            // test zero-dim
            array1 = manager.create(new Shape(4, 2, 1, 0));
            array2 = manager.create(new Shape(4, 2, 1, 0));
            array1.copyTo(array2);
            Assert.assertEquals(array1, array2, "CopyTo NDArray failed");
        }
    }

    @Test
    public void testNonZero() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.ones(new Shape(3, 3));
            NDArray actual =
                    manager.create(
                            new long[] {0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2},
                            new Shape(9, 2));
            Assert.assertEquals(actual, array.nonzero());
            // test multi-dim
            array = manager.create(new float[] {0f, 1f, 2f, 0f, 0f, -4f}, new Shape(2, 1, 3));
            actual = manager.create(new long[] {0, 0, 1, 0, 0, 2, 1, 0, 2}, new Shape(3, 3));
            Assert.assertEquals(actual, array.nonzero());
            // test scalar
            array = manager.create(1f);
            actual = manager.create(new long[] {0}, new Shape(1, 1));
            Assert.assertEquals(actual, array.nonzero());
            // test zero-dim
            // TODO confirm this behavior is right
            array = manager.create(new Shape(4, 0));
            actual = manager.create(new Shape(0, 2), DataType.INT64);
            Assert.assertEquals(actual, array.nonzero());
        }
    }

    @Test
    public void testAll() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10);
            Assert.assertFalse(array.all().getBoolean());
            Assert.assertTrue(array.add(1f).all().getBoolean());
            array = manager.create(new boolean[] {true, false});
            Assert.assertFalse(array.all().getBoolean());

            // test multi-dim
            array = manager.arange(20).reshape(2, 5, 2);
            Assert.assertFalse(array.all().getBoolean());
            Assert.assertTrue(array.add(1f).all().getBoolean());
            array =
                    manager.create(
                            new boolean[] {true, false, true, false, true, false}, new Shape(2, 3));
            Assert.assertFalse(array.all().getBoolean());

            // test scalar
            array = manager.create(1f);
            Assert.assertTrue(array.all().getBoolean());
            array = manager.create(false);
            Assert.assertFalse(array.all().getBoolean());

            // test zero-dim
            array = manager.create(new Shape(0));
            Assert.assertTrue(array.all().getBoolean());
        }
    }

    @Test
    public void testAny() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10);
            Assert.assertTrue(array.any().getBoolean());
            array = manager.zeros(new Shape(10));
            Assert.assertFalse(array.all().getBoolean());
            array = manager.create(new boolean[] {true, false});
            Assert.assertTrue(array.any().getBoolean());

            // test multi-dim
            array = manager.eye(2);
            Assert.assertTrue(array.any().getBoolean());
            array =
                    manager.create(
                            new boolean[] {true, false, true, false, true, false}, new Shape(2, 3));
            Assert.assertTrue(array.any().getBoolean());

            // test scalar
            array = manager.create(1f);
            Assert.assertTrue(array.any().getBoolean());
            array = manager.create(false);
            Assert.assertFalse(array.any().getBoolean());

            // test zero-dim
            array = manager.create(new Shape(0));
            Assert.assertFalse(array.any().getBoolean());
        }
    }

    @Test
    public void testNone() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10);
            Assert.assertFalse(array.none().getBoolean());
            array = manager.zeros(new Shape(10));
            Assert.assertTrue(array.none().getBoolean());
            array = manager.create(new boolean[] {false, false});
            Assert.assertTrue(array.none().getBoolean());

            // test multi-dim
            array = manager.eye(2);
            Assert.assertFalse(array.none().getBoolean());
            array =
                    manager.create(
                            new boolean[] {false, false, false, false, false, false},
                            new Shape(2, 3));
            Assert.assertTrue(array.none().getBoolean());

            // test scalar
            array = manager.create(1f);
            Assert.assertFalse(array.none().getBoolean());
            array = manager.create(false);
            Assert.assertTrue(array.none().getBoolean());

            // test zero-dim
            array = manager.create(new Shape(0));
            Assert.assertTrue(array.none().getBoolean());
        }
    }

    @Test
    public void testCountNonzero() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(4);
            Assert.assertEquals(3, array.countNonzero().getLong());

            // multi-dim
            array = manager.create(new float[] {-1f, 0f, 2f, 100f, 2340f, -200f}, new Shape(2, 3));
            Assert.assertEquals(5, array.countNonzero().getLong());

            // scalar
            array = manager.create(5f);
            Assert.assertEquals(1, array.countNonzero().getLong());
            // zero-dim
            array = manager.create(new Shape(2, 0));
            Assert.assertEquals(0, array.countNonzero().getLong());
        }
    }

    @Test
    public void testIsNaN() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {Float.NaN, 0f});
            NDArray actual = manager.create(new boolean[] {true, false});
            Assert.assertEquals(actual, array.isNaN());
            array = manager.create(new float[] {1f, 2f});
            Assert.assertFalse(array.isNaN().all().getBoolean());

            // test multi-dim
            array =
                    manager.create(
                            new float[] {Float.NaN, Float.NaN, Float.NaN, 0f}, new Shape(2, 2));
            actual = manager.create(new boolean[] {true, true, true, false}, new Shape(2, 2));
            Assert.assertEquals(actual, array.isNaN());

            // test scalar
            array = manager.create(Float.NaN);
            actual = manager.create(true);
            Assert.assertEquals(actual, array.isNaN());

            // test zero-dim
            array = manager.create(new Shape(0));
            actual = manager.create(new Shape(0), DataType.BOOLEAN);
            Assert.assertEquals(actual, array.isNaN());
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testBooleanMask() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(4);
            NDArray index = manager.create(new boolean[] {true, false, true, false});
            NDArray actual = manager.create(new float[] {0f, 2f});
            Assert.assertEquals(actual, array.booleanMask(index));
            Assert.assertEquals(actual, NDArrays.booleanMask(array, index));

            // test multi-dim
            array = manager.arange(10).reshape(2, 1, 5);
            index = manager.create(new boolean[] {true, false});
            actual = manager.arange(5).reshape(1, 1, 5);
            Assert.assertEquals(actual, array.booleanMask(index));
            Assert.assertEquals(actual, NDArrays.booleanMask(array, index));

            // test scalar
            array = manager.create(5f);
            index = manager.create(true);
            array.booleanMask(index);

            // test zero-dim
            array = manager.create(new Shape(1, 0));
            index = manager.create(new boolean[] {false});
            actual = manager.create(new Shape(0, 0));
            Assert.assertEquals(actual, array.booleanMask(index));
            Assert.assertEquals(actual, NDArrays.booleanMask(array, index));
        }
    }

    @Test
    public void testArgSort() {
        // TODO switch to numpy argsort
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {-1f, 2f, 0f, 999f, -998f});
            NDArray actual = manager.create(new int[] {4, 0, 2, 1, 3});
            Assert.assertEquals(actual, array.argSort());
            // multi-dim
            array =
                    manager.create(
                            new float[] {-1.000f, -0.009f, -0.0001f, 0.0001f, 0.12f, 0.1201f},
                            new Shape(2, 1, 1, 3, 1));
            actual = manager.zeros(new Shape(2, 1, 1, 3, 1), DataType.INT32);
            Assert.assertEquals(actual, array.argSort());
            // test axis
            array = manager.arange(10).reshape(2, 1, 5);
            actual = manager.create(new int[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new Shape(2, 1, 5));
            Assert.assertEquals(actual, array.argSort(0));
            actual = manager.zeros(new Shape(2, 1, 5), DataType.INT32);
            Assert.assertEquals(actual, array.argSort(1));
            actual = manager.create(new int[] {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}, new Shape(2, 1, 5));
            Assert.assertEquals(actual, array.argSort(2));
        }
    }

    @Test
    public void testSort() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {2f, 1f, 4f, 3f});
            NDArray actual = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assert.assertEquals(actual, array.sort());
            // test multi-dim
            array =
                    manager.create(
                            new float[] {1.01f, 0.00f, 0.01f, -0.05f, 1.0f, 0.9f, 0.99f, 0.999f},
                            new Shape(2, 2, 1, 2));
            actual =
                    manager.create(
                            new float[] {0f, 1.01f, -0.05f, 0.01f, 0.9f, 1f, 0.99f, 0.999f},
                            new Shape(2, 2, 1, 2));
            Assert.assertEquals(actual, array.sort());
            // test axis
            array =
                    manager.create(
                            new float[] {0f, 2f, 4f, 6f, 7f, 5f, 3f, 1f}, new Shape(2, 1, 2, 2));
            actual =
                    manager.create(
                            new float[] {0f, 2f, 3f, 1f, 7f, 5f, 4f, 6f}, new Shape(2, 1, 2, 2));
            Assert.assertEquals(actual, array.sort(0));
            actual =
                    manager.create(
                            new float[] {0f, 2f, 4f, 6f, 7f, 5f, 3f, 1f}, new Shape(2, 1, 2, 2));
            Assert.assertEquals(actual, array.sort(1));
            actual =
                    manager.create(
                            new float[] {0f, 2f, 4f, 6f, 3f, 1f, 7f, 5f}, new Shape(2, 1, 2, 2));
            Assert.assertEquals(actual, array.sort(2));
            actual =
                    manager.create(
                            new float[] {0f, 2f, 4f, 6f, 5f, 7f, 1f, 3f}, new Shape(2, 1, 2, 2));
            Assert.assertEquals(actual, array.sort(3));

            // scalar
            array = manager.create(5f);
            Assert.assertEquals(array, array.sort());

            // zero-dim
            array = manager.create(new Shape(1, 0, 1));
            Assert.assertEquals(array, array.sort());
            Assert.assertEquals(array, array.sort(0));
            Assert.assertEquals(array, array.sort(1));
            Assert.assertEquals(array, array.sort(2));
        }
    }

    @Test
    public void testSoftmax() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.ones(new Shape(10));
            NDArray actual = manager.zeros(new Shape(10)).add(0.1f);
            Assert.assertEquals(actual, array.softmax(0));
            // test multi-dim
            array = manager.ones(new Shape(2, 3, 1, 3));
            actual = manager.zeros(new Shape(2, 3, 1, 3)).add(0.5f);
            Assert.assertEquals(actual, array.softmax(0));
            actual = manager.zeros(new Shape(2, 3, 1, 3)).add(0.33333334f);
            Assert.assertEquals(actual, array.softmax(1));
            actual = manager.ones(new Shape(2, 3, 1, 3));
            Assert.assertEquals(actual, array.softmax(2));
            actual = manager.zeros(new Shape(2, 3, 1, 3)).add(0.33333334f);
            Assert.assertEquals(actual, array.softmax(3));
            // test scalar
            array = manager.create(1f);
            Assert.assertEquals(array, array.softmax(0));
            // test zero
            array = manager.create(new Shape(2, 0, 1));
            Assert.assertEquals(array, array.softmax(0));
        }
    }

    @Test
    public void testCumsum() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10);
            NDArray actual =
                    manager.create(new float[] {0f, 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f});
            Assert.assertEquals(actual, array.cumSum());

            array = manager.create(new float[] {1f, 2f, 3f, 5f, 8f, 13f});
            actual = manager.create(new float[] {1f, 3f, 6f, 11f, 19f, 32f});
            Assert.assertEquals(actual, array.cumSum(0));

            // test multi-dim
            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 7f, 9f, 11f, 13f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumSum(0));

            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumSum(1));

            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 3f, 6f, 10f, 5f, 11f, 18f, 26f, 35f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumSum(2));

            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumSum(3));

            // Note that shape after cumsum op with zero-dim and scalar case change
            // test scalar
            array = manager.create(1f);
            actual = manager.create(new float[] {1f});
            Assert.assertEquals(actual, array.cumSum());
            // test zero-dim
            array = manager.create(new Shape(2, 0));
            actual = manager.create(new Shape(0));
            Assert.assertEquals(actual, array.cumSum());
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testTile() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));

            NDArray tileAll = array.tile(2);
            NDArray tileAllActual =
                    manager.create(
                            new float[] {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4},
                            new Shape(4, 4));
            Assert.assertEquals(tileAllActual, tileAll, "Incorrect tile all");

            NDArray tileAxis = array.tile(0, 3);
            NDArray tileAxisActual =
                    manager.create(
                            new float[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, new Shape(6, 2));
            Assert.assertEquals(tileAxisActual, tileAxis, "Incorrect tile on axis");

            NDArray tileArray = array.tile(new long[] {3, 1});
            Assert.assertEquals(tileAxisActual, tileArray, "Incorrect tile array");

            NDArray tileShape = array.tile(new Shape(4));
            NDArray tileShapeActual =
                    manager.create(new float[] {1, 2, 1, 2, 3, 4, 3, 4}, new Shape(2, 4));
            Assert.assertEquals(tileShapeActual, tileShape, "Incorrect tile shape");

            // scalar
            array = manager.create(5f);

            tileAxis = manager.create(new float[] {5f, 5f, 5f}, new Shape(1, 3));
            Assert.assertEquals(tileAxis, array.tile(1, 3));

            // zero-dim
            array = manager.create(new Shape(2, 0));
            tileAllActual = manager.create(new Shape(2, 0));
            Assert.assertEquals(tileAllActual, array.tile(5));
            tileAllActual = manager.create(new Shape(10, 0));
            Assert.assertEquals(tileAllActual, array.tile(0, 5));

            array.tile(new Shape(2, 2, 2));
        }
    }

    @Test
    public void testRepeat() {
        // TODO add scalar and zero-dim test cases after fix the bug in MXNet np.repeat
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));

            NDArray repeatAll = array.repeat(2);
            NDArray repeatAllActual =
                    manager.create(
                            new float[] {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4},
                            new Shape(4, 4));
            Assert.assertEquals(repeatAllActual, repeatAll, "Incorrect repeat all");

            NDArray repeatAxis = array.repeat(0, 3);
            NDArray repeatAxisActual =
                    manager.create(
                            new float[] {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}, new Shape(6, 2));
            Assert.assertEquals(repeatAxisActual, repeatAxis, "Incorrect repeat on axis");

            NDArray repeatArray = array.repeat(new long[] {3, 1});
            Assert.assertEquals(repeatAxisActual, repeatArray, "Incorrect repeat array");

            NDArray repeatShape = array.repeat(new Shape(4));
            NDArray repeatShapeActual =
                    manager.create(new float[] {1, 1, 2, 2, 3, 3, 4, 4}, new Shape(2, 4));
            Assert.assertEquals(repeatShapeActual, repeatShape, "Incorrect repeat shape");
        }
    }

    @Test
    public void testClip() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray actual = manager.create(new float[] {2f, 2f, 3f, 3f});
            Assert.assertEquals(actual, original.clip(2.0, 3.0));
            Assert.assertEquals(actual, original.clip(2, 3));
            // multi-dim
            original =
                    manager.create(new float[] {5f, 4f, 2f, 5f, 6f, 7f, 2f, 22f, -23f, -2f})
                            .reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {3f, 3f, 2f, 3f, 3f, 3f, 2f, 3f, 2f, 2f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, original.clip(2.0, 3.0));
            Assert.assertEquals(actual, original.clip(2, 3));

            // scalar
            original = manager.create(5f);
            actual = manager.create(1f);
            Assert.assertEquals(actual, original.clip(0.0, 1.0));
            Assert.assertEquals(actual, original.clip(0, 1));
            // zero-dim
            original = manager.create(new Shape(0, 0));
            Assert.assertEquals(original, original.clip(0.0, 1.0));
            Assert.assertEquals(original, original.clip(0, 1));
        }
    }

    @Test(expectedExceptions = EngineException.class)
    public void testSwapAxes() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10).reshape(new Shape(2, 5));
            NDArray actual =
                    manager.create(new float[] {0, 5, 1, 6, 2, 7, 3, 8, 4, 9}, new Shape(5, 2));
            Assert.assertEquals(actual, array.swapAxes(0, 1));

            // TODO MXNet engine crash
            // scalar
            // array = manager.create(5f);
            // array.swapaxes(0, 1);

            // test zero-dim
            array = manager.create(new Shape(2, 0));
            actual = manager.create(new Shape(0, 2));
            Assert.assertEquals(actual, array.swapAxes(0, 1));
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testTranspose() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 2, 2));

            NDArray transposeAll = original.transpose();
            NDArray transposeAllActual =
                    manager.create(new float[] {1, 3, 2, 4}, new Shape(2, 2, 1));
            Assert.assertEquals(transposeAllActual, transposeAll, "Incorrect transpose all");

            NDArray transpose = original.transpose(1, 0, 2);
            NDArray transposeActual = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 1, 2));
            Assert.assertEquals(transposeActual, transpose, "Incorrect transpose all");
            Assert.assertEquals(transposeActual, original.swapAxes(0, 1), "Incorrect swap axes");

            // scalar
            original = manager.create(5f);
            Assert.assertEquals(original, original.transpose());
            original.transpose(0);

            // zero-dim
            original = manager.create(new Shape(2, 0, 1));
            transposeActual = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(transposeActual, original.transpose());
            transposeActual = manager.create(new Shape(2, 1, 0));
            Assert.assertEquals(transposeActual, original.transpose(0, 2, 1));
        }
    }

    @Test
    public void testBroadcast() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1, 2});
            NDArray broadcasted = array.broadcast(2, 2);
            NDArray actual = manager.create(new float[] {1, 2, 1, 2}, new Shape(2, 2));
            Assert.assertEquals(actual, broadcasted);
            // multi-dim
            array = manager.arange(4).reshape(2, 2);
            broadcasted = array.broadcast(3, 2, 2);
            actual = manager.arange(4).reshape(2, 2);
            actual = NDArrays.stack(new NDList(actual, actual, actual));
            Assert.assertEquals(actual, broadcasted);
            // scalar
            array = manager.create(1f);
            broadcasted = array.broadcast(2, 3, 2);
            actual = manager.ones(new Shape(2, 3, 2));
            Assert.assertEquals(actual, broadcasted);
            // zero-dim
            array = manager.create(new Shape(2, 0, 1));
            broadcasted = array.broadcast(2, 2, 0, 2);
            actual = manager.create(new Shape(2, 2, 0, 2));
            Assert.assertEquals(actual, broadcasted);
        }
    }

    @Test(expectedExceptions = EngineException.class)
    public void testArgMax() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array =
                    manager.create(
                            new float[] {
                                1, 2, 3, 4, 4, 5, 6, 23, 54, 234, 54, 23, 54, 4, 34, 34, 23, 54, 4,
                                3
                            },
                            new Shape(4, 5));
            NDArray argMax = array.argMax();
            NDArray actual = manager.create(9f);
            Assert.assertEquals(actual, argMax, "Argmax: Incorrect value");

            argMax = array.argMax(0);
            actual = manager.create(new float[] {2, 2, 2, 1, 1});
            Assert.assertEquals(actual, argMax, "Argmax: Incorrect value");

            argMax = array.argMax(1);
            actual = manager.create(new float[] {3, 4, 0, 2});
            Assert.assertEquals(actual, argMax, "Argmax: Incorrect value");

            // scalar
            array = manager.create(5f);
            // TODO the dtype should be int instead of float
            // Bug in MXNet to fix
            actual = manager.create(0f);
            Assert.assertEquals(actual, array.argMax());
            Assert.assertEquals(actual, array.argMax(0));

            // TODO MXNet engine crash
            // zero-dim
            // array = manager.create(new Shape(2, 0, 1));
            // array.argMax();
        }
    }

    @Test
    public void testArgMin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array =
                    manager.create(
                            new float[] {
                                1, 23, 3, 74, 4, 5, 6, -23, -54, 234, 54, 2, 54, 4, -34, 34, 23,
                                -54, 4, 3
                            },
                            new Shape(4, 5));
            NDArray argMin = array.argMin();
            NDArray actual = manager.create(8f);
            Assert.assertEquals(actual, argMin, "ArgMin: Incorrect value");

            argMin = array.argMin(0);
            actual = manager.create(new float[] {0, 2, 3, 1, 2});
            Assert.assertEquals(actual, argMin, "ArgMin: Incorrect value");

            argMin = array.argMin(1);
            actual = manager.create(new float[] {0, 3, 4, 2});
            Assert.assertEquals(actual, argMin, "ArgMin: Incorrect value");

            // scalar
            array = manager.create(1f);
            actual = manager.create(0f);
            Assert.assertEquals(actual, array.argMin(), "ArgMin: Incorrect value");
            Assert.assertEquals(actual, array.argMin(0), "ArgMin: Incorrect value");

            // zero-dim
            array = manager.create(new Shape(0, 1, 0));
            actual = manager.create(new Shape(0, 0));
            Assert.assertEquals(actual, array.argMin(1), "ArgMin: Incorrect value");
        }
    }
}
