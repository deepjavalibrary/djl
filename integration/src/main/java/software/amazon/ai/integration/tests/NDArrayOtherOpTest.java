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

import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayOtherOpTest extends AbstractTest {
    NDManager manager = NDManager.newBaseManager();

    public static void main(String[] args) {
        new NDArrayOtherOpTest().runTest(args);
    }

    @RunAsTest
    public void testGet() throws FailedTestException {
        NDArray original = manager.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});
        Assertions.assertEquals(original.get(new NDIndex()), original);

        NDArray getAt = original.get(0);
        NDArray getAtExpected = manager.create(new Shape(2), new float[] {1f, 2f});
        Assertions.assertEquals(getAt, getAtExpected);

        NDArray getSlice = original.get("1:");
        NDArray getSliceExpected = manager.create(new Shape(1, 2), new float[] {3f, 4f});
        Assertions.assertEquals(getSlice, getSliceExpected);
    }

    @RunAsTest
    public void testCopyTo() throws FailedTestException {
        NDArray ndArray1 = manager.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = manager.create(new DataDesc(new Shape(1, 4)));
        ndArray1.copyTo(ndArray2);
        ndArray1.contentEquals(ndArray2);
        Assertions.assertEquals(ndArray1, ndArray2, "CopyTo NDArray failed");
    }

    @RunAsTest
    public void testNonZero() throws FailedTestException {
        NDArray ndArray1 = manager.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = manager.create(new Shape(1, 4), new float[] {1f, 2f, 0f, 4f});
        NDArray ndArray3 = manager.create(new Shape(1, 4), new float[] {0f, 0f, 0f, 4f});
        NDArray ndArray4 = manager.create(new Shape(1, 4), new float[] {0f, 0f, 0f, 0f});
        Assertions.assertTrue(
                ndArray1.nonzero() == 4
                        && ndArray2.nonzero() == 3
                        && ndArray3.nonzero() == 1
                        && ndArray4.nonzero() == 0,
                "nonzero function returned incorrect value");
    }

    @RunAsTest
    public void testArgsort() throws FailedTestException {}

    @RunAsTest
    public void testSort() throws FailedTestException {
        NDArray original = manager.create(new Shape(2, 2), new float[] {2f, 1f, 4f, 3f});
        NDArray expected = manager.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});
        Assertions.assertEquals(original.sort(), expected);
    }

    @RunAsTest
    public void testSoftmax() throws FailedTestException {}

    @RunAsTest
    public void testCumsum() throws FailedTestException {
        NDArray expectedND =
                manager.create(new Shape(10), new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
        NDArray actualND =
                manager.create(
                        new Shape(10), new float[] {0f, 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f});
        Assertions.assertEquals(expectedND.cumsum(0), actualND);
    }

    @RunAsTest
    public void testCumsumi() throws FailedTestException {
        NDArray expectedND =
                manager.create(new Shape(10), new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
        NDArray actualND =
                manager.create(
                        new Shape(10), new float[] {0f, 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f});
        Assertions.assertEquals(expectedND.cumsumi(0), actualND);
        Assertions.assertInPlace(expectedND.cumsumi(0), expectedND);
    }

    @RunAsTest
    public void testTile() throws FailedTestException {
        NDArray original = manager.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});

        NDArray tileAll = original.tile(2);
        NDArray tileAllExpected =
                manager.create(
                        new Shape(4, 4),
                        new float[] {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4});
        Assertions.assertEquals(tileAll, tileAllExpected, "Incorrect tile all");

        NDArray tileAxis = original.tile(0, 3);
        NDArray tileAxisExpected =
                manager.create(new Shape(6, 2), new float[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
        Assertions.assertEquals(tileAxis, tileAxisExpected, "Incorrect tile on axis");

        NDArray tileArray = original.tile(new long[] {3, 1});
        Assertions.assertTrue(tileArray.contentEquals(tileAxisExpected), "Incorrect tile array");

        NDArray tileShape = original.tile(new Shape(4));
        NDArray tileShapeExpected =
                manager.create(new Shape(2, 4), new float[] {1, 2, 1, 2, 3, 4, 3, 4});
        Assertions.assertEquals(tileShape, tileShapeExpected, "Incorrect tile shape");
    }

    @RunAsTest
    public void testRepeat() throws FailedTestException {
        NDArray original = manager.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});

        NDArray repeatAll = original.repeat(2);
        NDArray repeatAllExpected =
                manager.create(
                        new Shape(4, 4),
                        new float[] {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
        Assertions.assertEquals(repeatAll, repeatAllExpected, "Incorrect repeat all");

        NDArray repeatAxis = original.repeat(0, 3);
        NDArray repeatAxisExpected =
                manager.create(new Shape(6, 2), new float[] {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4});
        Assertions.assertEquals(repeatAxis, repeatAxisExpected, "Incorrect repeat on axis");

        NDArray repeatArray = original.repeat(new long[] {3, 1});
        Assertions.assertEquals(repeatArray, repeatAxisExpected, "Incorrect repeat array");

        NDArray repeatShape = original.repeat(new Shape(4));
        NDArray repeatShapeExpected =
                manager.create(new Shape(2, 4), new float[] {1, 1, 2, 2, 3, 3, 4, 4});
        Assertions.assertEquals(repeatShape, repeatShapeExpected, "Incorrect repeat shape");
    }

    @RunAsTest
    public void testClip() throws FailedTestException {
        NDArray original = manager.create(new Shape(4), new float[] {1f, 2f, 3f, 4f});
        NDArray actual = manager.create(new Shape(4), new float[] {2f, 2f, 3f, 3f});

        Assertions.assertEquals(original.clip(2.0, 3.0), actual);
    }

    @RunAsTest
    public void testTranspose() throws FailedTestException {
        NDArray original = manager.create(new Shape(1, 2, 2), new float[] {1f, 2f, 3f, 4f});

        NDArray transposeAll = original.transpose();
        NDArray transposeAllExpected = manager.create(new Shape(2, 2, 1), new float[] {1, 3, 2, 4});
        Assertions.assertEquals(transposeAll, transposeAllExpected, "Incorrect transpose all");

        NDArray transpose = original.transpose(new int[] {1, 0, 2});
        NDArray transposeExpected = manager.create(new Shape(2, 1, 2), new float[] {1, 2, 3, 4});
        Assertions.assertEquals(transpose, transposeExpected, "Incorrect transpose all");
        Assertions.assertEquals(original.swapAxes(0, 1), transposeExpected, "Incorrect swap axes");
    }

    @RunAsTest
    public void testArgMax() throws FailedTestException {
        NDArray original =
                manager.create(
                        new Shape(4, 5),
                        new float[] {
                            1, 2, 3, 4, 4, 5, 6, 23, 54, 234, 54, 23, 54, 4, 34, 34, 23, 54, 4, 3
                        });
        NDArray argMax = original.argMax();
        NDArray expected = manager.create(new Shape(1), new float[] {9});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMax(0, true);
        expected = manager.create(new Shape(1, 5), new float[] {2, 2, 2, 1, 1});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMax(1, false);
        expected = manager.create(new Shape(4), new float[] {3, 4, 0, 2});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");
    }

    @RunAsTest
    public void testArgMin() throws FailedTestException {
        NDArray original =
                manager.create(
                        new Shape(4, 5),
                        new float[] {
                            1, 23, 3, 74, 4, 5, 6, -23, -54, 234, 54, 2, 54, 4, -34, 34, 23, -54, 4,
                            3
                        });
        NDArray argMax = original.argMin();
        NDArray expected = manager.create(new Shape(1), new float[] {8});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMin(0, false);
        expected = manager.create(new Shape(5), new float[] {0, 2, 3, 1, 2});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMin(1, true);
        expected = manager.create(new Shape(4, 1), new float[] {0, 3, 4, 2});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");
    }

    @RunAsTest
    public void testMatrixMultiplication() throws FailedTestException {
        NDArray lhs = manager.create(new Shape(2, 3), new float[] {6, -9, -12, 15, 0, 4});
        NDArray rhs = manager.create(new Shape(3, 1), new float[] {2, 3, -4});
        NDArray result;
        lhs.attachGrad();
        try (MxAutograd autograd = new MxAutograd()) {
            autograd.setRecording(true);
            result = NDArrays.mmul(lhs, rhs);
            autograd.backward((MxNDArray) result);
        }
        NDArray expected = manager.create(new Shape(2, 1), new float[] {33, 14});
        Assertions.assertEquals(
                expected, result, "Matrix multiplication: Incorrect value in result ndarray");

        NDArray expectedGradient =
                manager.create(new Shape(2, 3), new float[] {2, 3, -4, 2, 3, -4});
        Assertions.assertEquals(
                expectedGradient,
                lhs.getGradient(),
                "Matrix multiplication: Incorrect gradient after backward");
    }

    @RunAsTest
    public void testLogicalNot() throws FailedTestException {
        double[] testedData = new double[] {-2., 0., 1.};
        NDArray testedND = manager.create(new Shape(testedData.length), testedData);
        double[] boolData = new double[] {0.0, 1.0, 0.0};
        NDArray expectedND = manager.create(new Shape(testedData.length), boolData);
        Assertions.assertAlmostEquals(testedND.logicalNot(), expectedND);
    }
}
