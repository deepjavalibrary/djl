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
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.Gradient;

public class NDArrayOtherOpTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", NDArrayOtherOpTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testGet() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            Assertions.assertEquals(original.get(new NDIndex()), original);

            NDArray getAt = original.get(0);
            NDArray getAtExpected = manager.create(new float[] {1f, 2f});
            Assertions.assertEquals(getAt, getAtExpected);

            Assertions.assertEquals(getAtExpected, original.get("0,:"));
            Assertions.assertEquals(getAtExpected, original.get("0,*"));

            NDArray getSlice = original.get("1:");
            NDArray getSliceExpected = manager.create(new float[] {3f, 4f}, new Shape(1, 2));
            Assertions.assertEquals(getSlice, getSliceExpected);

            NDArray getStepSlice = original.get("1::2");
            Assertions.assertEquals(getStepSlice, getSliceExpected);
        }
    }

    @RunAsTest
    public void testSetArray() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {9, 10, 3, 4}, new Shape(2, 2));
            NDArray value = manager.create(new float[] {9, 10});
            original.set(new NDIndex(0), value);
            Assertions.assertEquals(expected, original);
        }
    }

    @RunAsTest
    public void testSetArrayBroadcast() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2, 1));
            NDArray expected = manager.create(new float[] {9, 9, 3, 4}, new Shape(2, 2, 1));
            NDArray value = manager.create(new float[] {9});
            original.set(new NDIndex(0), value);
            Assertions.assertEquals(expected, original);
        }
    }

    @RunAsTest
    public void testSetNumber() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {9, 9, 3, 4}, new Shape(2, 2));
            original.set(new NDIndex(0), 9);
            Assertions.assertEquals(expected, original);
        }
    }

    @RunAsTest
    public void testSetScalar() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            original.setScalar(new NDIndex(0, 0), 0);
            Assertions.assertThrows(
                    () -> original.setScalar(new NDIndex(0), 1), IllegalArgumentException.class);
        }
    }

    @RunAsTest
    public void testCopyTo() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray ndArray1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 4));
            NDArray ndArray2 = manager.create(new DataDesc(new Shape(1, 4)));
            ndArray1.copyTo(ndArray2);
            ndArray1.contentEquals(ndArray2);
            Assertions.assertEquals(ndArray1, ndArray2, "CopyTo NDArray failed");
        }
    }

    @RunAsTest
    public void testNonZero() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray ndArray1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 4));
            NDArray ndArray2 = manager.create(new float[] {1f, 2f, 0f, 4f}, new Shape(1, 4));
            NDArray ndArray3 = manager.create(new float[] {0f, 0f, 0f, 4f}, new Shape(1, 4));
            NDArray ndArray4 = manager.create(new float[] {0f, 0f, 0f, 0f}, new Shape(1, 4));
            Assertions.assertTrue(
                    ndArray1.nonzero() == 4
                            && ndArray2.nonzero() == 3
                            && ndArray3.nonzero() == 1
                            && ndArray4.nonzero() == 0,
                    "nonzero function returned incorrect value");
        }
    }

    @RunAsTest
    public void testArgsort() {}

    @RunAsTest
    public void testSort() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {2f, 1f, 4f, 3f}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            Assertions.assertEquals(original.sort(), expected);
        }
    }

    @RunAsTest
    public void testSoftmax() {}

    @RunAsTest
    public void testCumsum() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray expectedND =
                    manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
            NDArray actualND =
                    manager.create(new float[] {0f, 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f});
            Assertions.assertEquals(expectedND.cumsum(0), actualND);
        }
    }

    @RunAsTest
    public void testCumsumi() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray expectedND =
                    manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
            NDArray actualND =
                    manager.create(new float[] {0f, 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f});
            Assertions.assertEquals(expectedND.cumsumi(0), actualND);
            Assertions.assertInPlace(expectedND.cumsumi(0), expectedND);
        }
    }

    @RunAsTest
    public void testTile() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));

            NDArray tileAll = original.tile(2);
            NDArray tileAllExpected =
                    manager.create(
                            new float[] {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4},
                            new Shape(4, 4));
            Assertions.assertEquals(tileAll, tileAllExpected, "Incorrect tile all");

            NDArray tileAxis = original.tile(0, 3);
            NDArray tileAxisExpected =
                    manager.create(
                            new float[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, new Shape(6, 2));
            Assertions.assertEquals(tileAxis, tileAxisExpected, "Incorrect tile on axis");

            NDArray tileArray = original.tile(new long[] {3, 1});
            Assertions.assertTrue(
                    tileArray.contentEquals(tileAxisExpected), "Incorrect tile array");

            NDArray tileShape = original.tile(new Shape(4));
            NDArray tileShapeExpected =
                    manager.create(new float[] {1, 2, 1, 2, 3, 4, 3, 4}, new Shape(2, 4));
            Assertions.assertEquals(tileShape, tileShapeExpected, "Incorrect tile shape");
        }
    }

    @RunAsTest
    public void testRepeat() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));

            NDArray repeatAll = original.repeat(2);
            NDArray repeatAllExpected =
                    manager.create(
                            new float[] {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4},
                            new Shape(4, 4));
            Assertions.assertEquals(repeatAll, repeatAllExpected, "Incorrect repeat all");

            NDArray repeatAxis = original.repeat(0, 3);
            NDArray repeatAxisExpected =
                    manager.create(
                            new float[] {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}, new Shape(6, 2));
            Assertions.assertEquals(repeatAxis, repeatAxisExpected, "Incorrect repeat on axis");

            NDArray repeatArray = original.repeat(new long[] {3, 1});
            Assertions.assertEquals(repeatArray, repeatAxisExpected, "Incorrect repeat array");

            NDArray repeatShape = original.repeat(new Shape(4));
            NDArray repeatShapeExpected =
                    manager.create(new float[] {1, 1, 2, 2, 3, 3, 4, 4}, new Shape(2, 4));
            Assertions.assertEquals(repeatShape, repeatShapeExpected, "Incorrect repeat shape");
        }
    }

    @RunAsTest
    public void testClip() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray actual = manager.create(new float[] {2f, 2f, 3f, 3f});
            Assertions.assertEquals(original.clip(2.0, 3.0), actual);
        }
    }

    @RunAsTest
    public void testSwapAxes() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(10).reshape(new Shape(2, 5));
            NDArray actual =
                    manager.create(new float[] {0, 5, 1, 6, 2, 7, 3, 8, 4, 9}, new Shape(5, 2));
            Assertions.assertEquals(original.swapAxes(0, 1), actual);
        }
    }

    @RunAsTest
    public void testTranspose() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 2, 2));

            NDArray transposeAll = original.transpose();
            NDArray transposeAllExpected =
                    manager.create(new float[] {1, 3, 2, 4}, new Shape(2, 2, 1));
            Assertions.assertEquals(transposeAll, transposeAllExpected, "Incorrect transpose all");

            NDArray transpose = original.transpose(new int[] {1, 0, 2});
            NDArray transposeExpected =
                    manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 1, 2));
            Assertions.assertEquals(transpose, transposeExpected, "Incorrect transpose all");
            Assertions.assertEquals(
                    original.swapAxes(0, 1), transposeExpected, "Incorrect swap axes");
        }
    }

    @RunAsTest
    public void testBroadcast() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2});

            NDArray broadcasted = original.broadcast(new Shape(2, 2));
            NDArray expected = manager.create(new float[] {1, 2, 1, 2}, new Shape(2, 2));
            Assertions.assertEquals(expected, broadcasted);
        }
    }

    @RunAsTest
    public void testArgmax() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {
                                1, 2, 3, 4, 4, 5, 6, 23, 54, 234, 54, 23, 54, 4, 34, 34, 23, 54, 4,
                                3
                            },
                            new Shape(4, 5));
            NDArray argMax = original.argmax();
            NDArray expected = manager.create(new float[] {9});
            Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

            argMax = original.argmax(0, true);
            expected = manager.create(new float[] {2, 2, 2, 1, 1}, new Shape(1, 5));
            Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

            argMax = original.argmax(1, false);
            expected = manager.create(new float[] {3, 4, 0, 2});
            Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");
        }
    }

    @RunAsTest
    public void testArgmin() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(
                            new float[] {
                                1, 23, 3, 74, 4, 5, 6, -23, -54, 234, 54, 2, 54, 4, -34, 34, 23,
                                -54, 4, 3
                            },
                            new Shape(4, 5));
            NDArray argMax = original.argmin();
            NDArray expected = manager.create(new float[] {8});
            Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

            argMax = original.argmin(0, false);
            expected = manager.create(new float[] {0, 2, 3, 1, 2});
            Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

            argMax = original.argmin(1, true);
            expected = manager.create(new float[] {0, 3, 4, 2}, new Shape(4, 1));
            Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");
        }
    }

    @RunAsTest
    public void testMatrixMultiplication() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            NDArray result;
            try (Gradient.Collector gradCol = Gradient.newCollector()) {
                Gradient.NDArrayKey lhsKey = gradCol.collectFor(lhs);
                result = NDArrays.mmul(lhs, rhs);
                Gradient.Dict grads = gradCol.collect(result);
                NDArray expected = manager.create(new float[] {33, 14}, new Shape(2, 1));
                Assertions.assertEquals(
                        expected,
                        result,
                        "Matrix multiplication: Incorrect value in result ndarray");

                NDArray expectedGradient =
                        manager.create(new float[] {2, 3, -4, 2, 3, -4}, new Shape(2, 3));
                Assertions.assertEquals(
                        expectedGradient,
                        grads.get(lhsKey).get(),
                        "Matrix multiplication: Incorrect gradient after backward");
            }
        }
    }
}
