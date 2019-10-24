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
import ai.djl.integration.util.Assertions;
import ai.djl.mxnet.engine.MxNDArray;
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
            NDArray ndArray1 = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray ndArray2 = manager.create(new float[] {1f, 2f, 0f, 4f});
            NDArray ndArray3 = manager.create(new float[] {0f, 0f, 0f, 4f});
            NDArray ndArray4 = manager.create(new float[] {0f, 0f, 0f, 0f});
            Assert.assertTrue(
                    ndArray1.nonzero() == 4
                            && ndArray2.nonzero() == 3
                            && ndArray3.nonzero() == 1
                            && ndArray4.nonzero() == 0,
                    "nonzero function returned incorrect value");
            // test multi-dim
            ndArray1 = manager.create(new float[] {0f, 1f, 2f, 0f, 0f, -4f}, new Shape(2, 1, 3));
            Assert.assertEquals(ndArray1.nonzero(), 3);
            // test scalar
            ndArray1 = manager.create(0f);
            ndArray2 = manager.create(10f);
            Assert.assertTrue(ndArray1.nonzero() == 0 && ndArray2.nonzero() == 1);
            // test zero-dim
            ndArray1 = manager.create(new Shape(4, 0));
            Assert.assertEquals(ndArray1.nonzero(), 0);
        }
    }

    @Test
    public void testArgsort() {
        // TODO switch to numpy argsort
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {-1f, 2f, 0f, 999f, -998f});
            NDArray actual = manager.create(new int[] {4, 0, 2, 1, 3});
            Assert.assertEquals(actual, array.argsort());
            // multi-dim
            array =
                    manager.create(
                            new float[] {-1.000f, -0.009f, -0.0001f, 0.0001f, 0.12f, 0.1201f},
                            new Shape(2, 1, 1, 3, 1));
            actual = manager.zeros(new Shape(2, 1, 1, 3, 1), DataType.INT32);
            Assert.assertEquals(actual, array.argsort());
            // test axis
            array = manager.arange(10).reshape(2, 1, 5);
            actual = manager.create(new int[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new Shape(2, 1, 5));
            Assert.assertEquals(actual, array.argsort(0));
            actual = manager.zeros(new Shape(2, 1, 5), DataType.INT32);
            Assert.assertEquals(actual, array.argsort(1));
            actual = manager.create(new int[] {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}, new Shape(2, 1, 5));
            Assert.assertEquals(actual, array.argsort(2));
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
            Assert.assertEquals(actual, array.cumsum());
            Assertions.assertInPlaceEquals(actual, array.cumsumi(), array);

            array = manager.create(new float[] {1f, 2f, 3f, 5f, 8f, 13f});
            actual = manager.create(new float[] {1f, 3f, 6f, 11f, 19f, 32f});
            Assert.assertEquals(actual, array.cumsum(0));
            Assertions.assertInPlaceEquals(actual, array.cumsumi(0), array);

            // test multi-dim
            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 7f, 9f, 11f, 13f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumsum(0));
            Assertions.assertInPlaceEquals(actual, array.cumsumi(0), array);

            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumsum(1));
            Assertions.assertInPlaceEquals(actual, array.cumsumi(1), array);

            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 3f, 6f, 10f, 5f, 11f, 18f, 26f, 35f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumsum(2));
            Assertions.assertInPlaceEquals(actual, array.cumsumi(2), array);

            array = manager.arange(10).reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, array.cumsum(3));
            Assertions.assertInPlaceEquals(actual, array.cumsumi(3), array);

            // Note that shape after cumsum op with zero-dim and scalar case change
            // test scalar
            array = manager.create(1f);
            actual = manager.create(new float[] {1f});
            Assert.assertEquals(actual, array.cumsum());
            // test zero-dim
            array = manager.create(new Shape(2, 0));
            actual = manager.create(new Shape(0));
            Assert.assertEquals(actual, array.cumsum());
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
            tileAllActual = manager.create(new float[] {5f, 5f, 5f});
            Assert.assertEquals(tileAllActual, array.tile(3));

            array.tile(0, 3);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testTileZeroDim() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // zero-dim
            NDArray array = manager.create(new Shape(2, 0));
            NDArray tileAllActual = manager.create(new Shape(2, 0));
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
            // multi-dim
            original =
                    manager.create(new float[] {5f, 4f, 2f, 5f, 6f, 7f, 2f, 22f, -23f, -2f})
                            .reshape(2, 1, 5, 1);
            actual =
                    manager.create(
                            new float[] {3f, 3f, 2f, 3f, 3f, 3f, 2f, 3f, 2f, 2f},
                            new Shape(2, 1, 5, 1));
            Assert.assertEquals(actual, original.clip(2.0, 3.0));
            // scalar
            original = manager.create(5f);
            actual = manager.create(1f);
            Assert.assertEquals(actual, original.clip(0.0, 1.0));
            // zero-dim
            original = manager.create(new Shape(0, 0));
            Assert.assertEquals(original, original.clip(0.0, 1.0));
        }
    }
    // TODO Enable after fixing failure in jenkins
    @Test(enabled = false, expectedExceptions = EngineException.class)
    public void testSwapAxes() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(10).reshape(new Shape(2, 5));
            NDArray actual =
                    manager.create(new float[] {0, 5, 1, 6, 2, 7, 3, 8, 4, 9}, new Shape(5, 2));
            Assert.assertEquals(actual, array.swapAxes(0, 1));
            // test zero-dim
            array = manager.create(new Shape(2, 0));
            actual = manager.create(new Shape(0, 2));
            Assert.assertEquals(actual, array.swapAxes(0, 1));

            // scalar
            array = manager.create(5f);
            array.swapAxes(0, 1);
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

            // zero-dim
            original = manager.create(new Shape(2, 0, 1));
            transposeActual = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(transposeActual, original.transpose());
            transposeActual = manager.create(new Shape(2, 1, 0));
            Assert.assertEquals(transposeActual, original.transpose(0, 2, 1));

            // scalar
            original = manager.create(5f);
            Assert.assertEquals(original, original.transpose());
            original.transpose(0);
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

    // TODO Enable after fixing failure in jenkins

    @Test(enabled = false, expectedExceptions = EngineException.class)
    public void testArgmax() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array =
                    manager.create(
                            new float[] {
                                1, 2, 3, 4, 4, 5, 6, 23, 54, 234, 54, 23, 54, 4, 34, 34, 23, 54, 4,
                                3
                            },
                            new Shape(4, 5));
            NDArray argmax = array.argmax();
            NDArray actual = manager.create(9f);
            Assert.assertEquals(actual, argmax, "Argmax: Incorrect value");

            argmax = array.argmax(0);
            actual = manager.create(new float[] {2, 2, 2, 1, 1});
            Assert.assertEquals(actual, argmax, "Argmax: Incorrect value");

            argmax = array.argmax(1);
            actual = manager.create(new float[] {3, 4, 0, 2});
            Assert.assertEquals(actual, argmax, "Argmax: Incorrect value");

            // scalar
            array = manager.create(5f);
            // TODO the dtype should be int instead of float
            // Bug in MXNet to fix
            actual = manager.create(0f);
            Assert.assertEquals(actual, array.argmax());
            Assert.assertEquals(actual, array.argmax(0));
            // zero-dim
            array = manager.create(new Shape(2, 0, 1));

            // add waitAll to make sure it catch the exception because
            // inferShape and operator forward happened in different threads

            ((MxNDArray) array.argmax()).waitAll();
        }
    }

    @Test
    public void testArgmin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array =
                    manager.create(
                            new float[] {
                                1, 23, 3, 74, 4, 5, 6, -23, -54, 234, 54, 2, 54, 4, -34, 34, 23,
                                -54, 4, 3
                            },
                            new Shape(4, 5));
            NDArray argmin = array.argmin();
            NDArray actual = manager.create(8f);
            Assert.assertEquals(actual, argmin, "Argmin: Incorrect value");

            argmin = array.argmin(0);
            actual = manager.create(new float[] {0, 2, 3, 1, 2});
            Assert.assertEquals(actual, argmin, "Argmin: Incorrect value");

            argmin = array.argmin(1);
            actual = manager.create(new float[] {0, 3, 4, 2});
            Assert.assertEquals(actual, argmin, "Argmin: Incorrect value");

            // scalar
            array = manager.create(1f);
            actual = manager.create(0f);
            Assert.assertEquals(actual, array.argmin(), "Argmin: Incorrect value");
            Assert.assertEquals(actual, array.argmin(0), "Argmin: Incorrect value");

            // zero-dim
            array = manager.create(new Shape(0, 1, 0));
            actual = manager.create(new Shape(0, 0));
            Assert.assertEquals(actual, array.argmin(1), "Argmin: Incorrect value");
        }
    }
}
