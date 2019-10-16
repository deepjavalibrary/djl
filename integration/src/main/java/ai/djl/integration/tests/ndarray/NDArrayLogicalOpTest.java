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

import ai.djl.integration.util.Assertions;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayLogicalOpTest {

    @Test
    public void testLogicalAnd() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.arange(10);
            NDArray array2 = manager.arange(10);
            NDArray actual = manager.create(new float[] {0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
            array1 = manager.arange(-5, 5).reshape(2, 5);
            array2 = manager.arange(5, -5, -1).reshape(2, 5);
            actual =
                    manager.create(
                            new float[] {1f, 1f, 1f, 1f, 1f, 0f, 1f, 1f, 1f, 1f}, new Shape(2, 5));
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
            // scalar
            array1 = manager.create(1f);
            array2 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            actual = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
            array2 = manager.create(0f);
            Assert.assertEquals(0f, array1.logicalAnd(array2).getFloat());
            Assert.assertEquals(0f, NDArrays.logicalAnd(array1, array2).getFloat());

            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array2, array1.logicalAnd(array2));
            Assert.assertEquals(array2, NDArrays.logicalAnd(array1, array2));
        }
    }

    @Test
    public void testLogicalOr() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.arange(10);
            NDArray array2 = manager.arange(10);
            NDArray actual = manager.create(new float[] {0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            array1 = manager.arange(-5, 5).reshape(5, 2);
            array2 = manager.arange(5, -5, -1).reshape(5, 2);
            actual =
                    manager.create(
                            new float[] {1f, 1f, 1f, 1f, 1f, 0f, 1f, 1f, 1f, 1f}, new Shape(5, 2));
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            // scalar
            array1 = manager.create(1f);
            array2 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            actual = manager.create(new float[] {1f, 1f, 1f, 1f}, new Shape(2, 2));
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array2, array1.logicalOr(array2));
            Assert.assertEquals(array2, NDArrays.logicalOr(array1, array2));
        }
    }

    @Test
    public void testLogicalXor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.arange(10);
            NDArray array2 = manager.arange(10);
            NDArray actual = manager.zeros(new Shape(10));
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
            array1 = manager.arange(-5, 5).reshape(2, 1, 5);
            array2 = manager.arange(5, -5, -1).reshape(2, 1, 5);
            actual = manager.zeros(new Shape(2, 1, 5));
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
            // scalar
            array1 = manager.create(1f);
            array2 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            actual = manager.create(new float[] {0f, 1f, 1f, 0f}, new Shape(2, 2));
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array2, array1.logicalXor(array2));
            Assert.assertEquals(array2, NDArrays.logicalXor(array1, array2));
        }
    }

    @Test
    public void testLogicalNot() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {-2f, 0f, 1f});
            NDArray actual = manager.create(new float[] {0f, 1f, 0f});
            Assertions.assertAlmostEquals(actual, array.logicalNot());
            array = manager.create(new float[] {1f, 2f, -1f, -2f}, new Shape(2, 2));
            actual = manager.create(new float[] {0f, 0f, 0f, 0f}, new Shape(2, 2));
            Assertions.assertAlmostEquals(actual, array.logicalNot());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(1f);
            Assert.assertEquals(actual, array.logicalNot());
            // test zero-dim
            array = manager.create(new Shape(0, 0, 1));
            Assert.assertEquals(array, array.logicalNot());
        }
    }
}
