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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayLogicalOpTest {

    @Test
    public void testLogicalAnd() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new boolean[] {true, false});
            NDArray array2 = manager.create(new boolean[] {false, false});
            NDArray actual = manager.create(new boolean[] {false, false});
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
            array1 = manager.arange(10);
            array2 = manager.arange(10);
            actual =
                    manager.create(
                            new boolean[] {
                                false, true, true, true, true, true, true, true, true, true
                            });
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));

            // test multi-dim
            array1 = manager.create(new boolean[] {true, true, false, false}, new Shape(2, 2));
            array2 = manager.create(new boolean[] {false, false, true, true}, new Shape(2, 2));
            actual = manager.create(new boolean[] {false, false, false, false}, new Shape(2, 2));
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
            array1 = manager.arange(-5, 5).reshape(2, 5);
            array2 = manager.arange(5, -5, -1).reshape(2, 5);
            actual =
                    manager.create(
                            new boolean[] {
                                true, true, true, true, true, false, true, true, true, true
                            },
                            new Shape(2, 5));
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
            // scalar
            array1 = manager.create(true);
            array2 = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            actual = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
            array2 = manager.create(false);
            actual = manager.create(false);
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));

            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            actual = manager.create(new Shape(1, 0, 2), DataType.BOOLEAN);
            Assert.assertEquals(actual, array1.logicalAnd(array2));
            Assert.assertEquals(actual, NDArrays.logicalAnd(array1, array2));
        }
    }

    @Test
    public void testLogicalOr() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new boolean[] {true, false, true, false});
            NDArray array2 = manager.create(new boolean[] {false, true, false, true});
            NDArray actual = manager.create(new boolean[] {true, true, true, true});
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            array1 = manager.arange(10);
            array2 = manager.arange(10);
            actual =
                    manager.create(
                            new boolean[] {
                                false, true, true, true, true, true, true, true, true, true
                            });
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            // test multi-dim
            array1 = manager.create(new boolean[] {false, false, false, false}, new Shape(4, 1));
            array2 = manager.create(new boolean[] {true, true, true, true}, new Shape(4, 1));
            actual = manager.create(new boolean[] {true, true, true, true}, new Shape(4, 1));
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            array1 = manager.arange(-5, 5).reshape(5, 2);
            array2 = manager.arange(5, -5, -1).reshape(5, 2);
            actual =
                    manager.create(
                            new boolean[] {
                                true, true, true, true, true, false, true, true, true, true
                            },
                            new Shape(5, 2));
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            // scalar
            array1 = manager.create(true);
            array2 = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            actual = manager.create(new boolean[] {true, true, true, true}, new Shape(2, 2));
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            actual = manager.create(new Shape(1, 0, 2), DataType.BOOLEAN);
            Assert.assertEquals(actual, array1.logicalOr(array2));
            Assert.assertEquals(actual, NDArrays.logicalOr(array1, array2));
        }
    }

    @Test
    public void testLogicalXor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new boolean[] {true, false, true, false});
            NDArray array2 = manager.create(new boolean[] {true, true, false, false});
            NDArray actual = manager.create(new boolean[] {false, true, true, false});
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
            array1 = manager.arange(10);
            array2 = manager.arange(10);
            actual = manager.zeros(new Shape(10)).asType(DataType.BOOLEAN, false);
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));

            // test multi-dim
            array1 = manager.create(new boolean[] {true, false, true, false}, new Shape(1, 4));
            array2 = manager.create(new boolean[] {true, true, false, false}, new Shape(1, 4));
            actual = manager.create(new boolean[] {false, true, true, false}, new Shape(1, 4));
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
            array1 = manager.arange(-5, 5).reshape(2, 1, 5);
            array2 = manager.arange(5, -5, -1).reshape(2, 1, 5);
            actual = manager.zeros(new Shape(2, 1, 5)).asType(DataType.BOOLEAN, false);
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
            // scalar
            array1 = manager.create(true);
            array2 = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            actual = manager.create(new boolean[] {false, true, true, false}, new Shape(2, 2));
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            actual = manager.create(new Shape(1, 0, 2), DataType.BOOLEAN);
            Assert.assertEquals(actual, array1.logicalXor(array2));
            Assert.assertEquals(actual, NDArrays.logicalXor(array1, array2));
        }
    }

    @Test
    public void testLogicalNot() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {-2f, 0f, 1f});
            NDArray actual = manager.create(new boolean[] {false, true, false});
            Assertions.assertAlmostEquals(actual, array.logicalNot());
            array = manager.create(new float[] {1f, 2f, -1f, -2f}, new Shape(2, 2));
            actual = manager.create(new boolean[] {false, false, false, false}, new Shape(2, 2));
            Assertions.assertAlmostEquals(actual, array.logicalNot());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(true);
            Assert.assertEquals(actual, array.logicalNot());
            // test zero-dim
            array = manager.create(new Shape(0, 0, 1));
            Assert.assertEquals(array.asType(DataType.BOOLEAN, false), array.logicalNot());
        }
    }
}
