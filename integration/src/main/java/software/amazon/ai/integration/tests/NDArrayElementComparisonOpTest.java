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

import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayElementComparisonOpTest extends AbstractTest {
    NDManager manager = NDManager.newBaseManager();

    public static void main(String[] args) {
        new NDArrayElementComparisonOpTest().runTest(args);
    }

    @RunAsTest
    public void testContentEquals() throws FailedTestException {
        NDArray original = manager.create(new Shape(2), new float[] {1f, 2f});
        NDArray expect = manager.create(new Shape(2), new float[] {1f, 2f});
        if (!original.contentEquals(expect)) {
            throw new FailedTestException("testContentEquals tests failed!");
        }
    }

    @RunAsTest
    public void testEqualsForEqualNDArray() throws FailedTestException {
        NDArray ndArray1 = manager.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = manager.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray result = NDArrays.eq(ndArray1, ndArray2);
        Assertions.assertTrue(
                result.nonzero() == 4 && NDArrays.equals(ndArray1, ndArray2),
                "Incorrect comparison for equal NDArray");
    }

    @RunAsTest
    public void testEqualsForScalar() throws FailedTestException {
        NDArray ndArray = manager.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray result = NDArrays.eq(ndArray, 2);
        Assertions.assertTrue(result.nonzero() == 1, "Incorrect comparison for equal NDArray");
    }

    @RunAsTest
    public void testEqualsForUnEqualNDArray() throws FailedTestException {
        NDArray ndArray1 = manager.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = manager.create(new Shape(1, 4), new float[] {1f, 3f, 3f, 4f});
        NDArray result = NDArrays.eq(ndArray1, ndArray2);
        Assertions.assertTrue(
                result.nonzero() == 3 && !NDArrays.equals(ndArray1, ndArray2),
                "Incorrect comparison for unequal NDArray");
    }

    @RunAsTest
    public void testGreaterThanScalar() throws FailedTestException {
        NDArray array = manager.create(new Shape(1, 5), new float[] {1, 0, 2f, 2f, 4f});
        NDArray greater = NDArrays.gt(array, 2);
        Assertions.assertTrue(greater.nonzero() == 1, "greater_scalar: Incorrect comparison");
    }

    @RunAsTest
    public void testGreaterThanOrEqualToScalar() throws FailedTestException {
        NDArray array = manager.create(new Shape(1, 4), new float[] {1f, 2f, 2f, 4f});
        NDArray greater = NDArrays.gte(array, 2);
        Assertions.assertTrue(
                greater.nonzero() == 3, "greater_equals_scalar: Incorrect comparison");
    }

    @RunAsTest
    public void testGreaterThanAndLessThan() throws FailedTestException {
        NDArray ndArray1 = manager.create(new Shape(1, 6), new float[] {1f, 2f, 2f, 4f, 5f, 4f});
        NDArray ndArray2 = manager.create(new Shape(1, 6), new float[] {2f, 1f, 2f, 5f, 4f, 5f});
        NDArray greater = NDArrays.gt(ndArray1, ndArray2);
        Assertions.assertTrue(greater.nonzero() == 2, "greater: Incorrect comparison");
        NDArray lesser = NDArrays.lt(ndArray1, ndArray2);
        Assertions.assertTrue(lesser.nonzero() == 3, "lesser: Incorrect comparison");
    }

    @RunAsTest
    public void testGreaterThanAndLessThanEquals() throws FailedTestException {
        NDArray ndArray1 = manager.create(new Shape(1, 6), new float[] {1f, 2f, 2f, 4f, 5f, 4f});
        NDArray ndArray2 = manager.create(new Shape(1, 6), new float[] {2f, 1f, 2f, 5f, 4f, 5f});
        NDArray greater = NDArrays.gte(ndArray1, ndArray2);
        Assertions.assertTrue(greater.nonzero() == 3, "greater_equal: Incorrect comparison");
        NDArray lesser = NDArrays.lte(ndArray1, ndArray2);
        Assertions.assertTrue(lesser.nonzero() == 4, "lesser_equal: Incorrect comparison");
    }

    @RunAsTest
    public void testLesserThanOrEqualToScalar() throws FailedTestException {
        NDArray array = manager.create(new Shape(1, 5), new float[] {1f, 2f, 2f, 4f, 5f});
        NDArray greater = NDArrays.lte(array, 2);
        Assertions.assertTrue(
                greater.nonzero() == 3, "lesser_equals_scalar: Incorrect comparison");
    }

    @RunAsTest
    public void testLesserThanScalar() throws FailedTestException {
        NDArray array = manager.create(new Shape(1, 5), new float[] {1f, 2f, 2f, 4f, 5f});
        NDArray greater = NDArrays.lt(array, 2);
        Assertions.assertTrue(greater.nonzero() == 1, "lesser_scalar: Incorrect comparison");
    }
}
