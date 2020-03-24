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
import ai.djl.testing.Assertions;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayLogicalOpTest {

    @Test
    public void testLogicalAnd() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new boolean[] {true, false});
            NDArray array2 = manager.create(new boolean[] {false, false});
            NDArray expected = manager.create(new boolean[] {false, false});
            Assert.assertEquals(array1.logicalAnd(array2), expected);
            Assert.assertEquals(NDArrays.logicalAnd(array1, array2), expected);
            array1 = manager.arange(10);
            array2 = manager.arange(10);
            expected =
                    manager.create(
                            new boolean[] {
                                false, true, true, true, true, true, true, true, true, true
                            });
            Assert.assertEquals(array1.logicalAnd(array2), expected);
            Assert.assertEquals(NDArrays.logicalAnd(array1, array2), expected);

            // test multi-dim
            array1 = manager.create(new boolean[] {true, true, false, false}, new Shape(2, 2));
            array2 = manager.create(new boolean[] {false, false, true, true}, new Shape(2, 2));
            expected = manager.create(new boolean[] {false, false, false, false}, new Shape(2, 2));
            Assert.assertEquals(array1.logicalAnd(array2), expected);
            Assert.assertEquals(NDArrays.logicalAnd(array1, array2), expected);
            array1 = manager.arange(-5.0f, 5.0f).reshape(2, 5);
            array2 = manager.arange(5.0f, -5.0f, -1.0f).reshape(2, 5);
            expected =
                    manager.create(
                            new boolean[] {
                                true, true, true, true, true, false, true, true, true, true
                            },
                            new Shape(2, 5));
            Assert.assertEquals(array1.logicalAnd(array2), expected);
            Assert.assertEquals(NDArrays.logicalAnd(array1, array2), expected);
            // scalar
            array1 = manager.create(true);
            array2 = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            expected = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            Assert.assertEquals(array1.logicalAnd(array2), expected);
            Assert.assertEquals(NDArrays.logicalAnd(array1, array2), expected);
            array2 = manager.create(false);
            expected = manager.create(false);
            Assert.assertEquals(array1.logicalAnd(array2), expected);
            Assert.assertEquals(NDArrays.logicalAnd(array1, array2), expected);

            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            expected = manager.create(new Shape(1, 0, 2), DataType.BOOLEAN);
            Assert.assertEquals(array1.logicalAnd(array2), expected);
            Assert.assertEquals(NDArrays.logicalAnd(array1, array2), expected);
        }
    }

    @Test
    public void testLogicalOr() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new boolean[] {true, false, true, false});
            NDArray array2 = manager.create(new boolean[] {false, true, false, true});
            NDArray expected = manager.create(new boolean[] {true, true, true, true});
            Assert.assertEquals(array1.logicalOr(array2), expected);
            Assert.assertEquals(NDArrays.logicalOr(array1, array2), expected);
            array1 = manager.arange(10.0f);
            array2 = manager.arange(10.0f);
            expected =
                    manager.create(
                            new boolean[] {
                                false, true, true, true, true, true, true, true, true, true
                            });
            Assert.assertEquals(array1.logicalOr(array2), expected);
            Assert.assertEquals(NDArrays.logicalOr(array1, array2), expected);
            // test multi-dim
            array1 = manager.create(new boolean[] {false, false, false, false}, new Shape(4, 1));
            array2 = manager.create(new boolean[] {true, true, true, true}, new Shape(4, 1));
            expected = manager.create(new boolean[] {true, true, true, true}, new Shape(4, 1));
            Assert.assertEquals(array1.logicalOr(array2), expected);
            Assert.assertEquals(NDArrays.logicalOr(array1, array2), expected);
            array1 = manager.arange(-5.0f, 5.0f).reshape(5, 2);
            array2 = manager.arange(5.0f, -5.0f, -1.0f).reshape(5, 2);
            expected =
                    manager.create(
                            new boolean[] {
                                true, true, true, true, true, false, true, true, true, true
                            },
                            new Shape(5, 2));
            Assert.assertEquals(array1.logicalOr(array2), expected);
            Assert.assertEquals(NDArrays.logicalOr(array1, array2), expected);
            // scalar
            array1 = manager.create(true);
            array2 = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            expected = manager.create(new boolean[] {true, true, true, true}, new Shape(2, 2));
            Assert.assertEquals(array1.logicalOr(array2), expected);
            Assert.assertEquals(NDArrays.logicalOr(array1, array2), expected);
            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            expected = manager.create(new Shape(1, 0, 2), DataType.BOOLEAN);
            Assert.assertEquals(array1.logicalOr(array2), expected);
            Assert.assertEquals(NDArrays.logicalOr(array1, array2), expected);
        }
    }

    @Test
    public void testLogicalXor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new boolean[] {true, false, true, false});
            NDArray array2 = manager.create(new boolean[] {true, true, false, false});
            NDArray expected = manager.create(new boolean[] {false, true, true, false});
            Assert.assertEquals(array1.logicalXor(array2), expected);
            Assert.assertEquals(NDArrays.logicalXor(array1, array2), expected);
            array1 = manager.arange(10.0f);
            array2 = manager.arange(10.0f);
            expected = manager.zeros(new Shape(10)).toType(DataType.BOOLEAN, false);
            Assert.assertEquals(array1.logicalXor(array2), expected);
            Assert.assertEquals(NDArrays.logicalXor(array1, array2), expected);

            // test multi-dim
            array1 = manager.create(new boolean[] {true, false, true, false}, new Shape(1, 4));
            array2 = manager.create(new boolean[] {true, true, false, false}, new Shape(1, 4));
            expected = manager.create(new boolean[] {false, true, true, false}, new Shape(1, 4));
            Assert.assertEquals(array1.logicalXor(array2), expected);
            Assert.assertEquals(NDArrays.logicalXor(array1, array2), expected);
            array1 = manager.arange(-5.0f, 5.0f).reshape(2, 1, 5);
            array2 = manager.arange(5.0f, -5.0f, -1.0f).reshape(2, 1, 5);
            expected = manager.zeros(new Shape(2, 1, 5)).toType(DataType.BOOLEAN, false);
            Assert.assertEquals(array1.logicalXor(array2), expected);
            Assert.assertEquals(NDArrays.logicalXor(array1, array2), expected);
            // scalar
            array1 = manager.create(true);
            array2 = manager.create(new boolean[] {true, false, false, true}, new Shape(2, 2));
            expected = manager.create(new boolean[] {false, true, true, false}, new Shape(2, 2));
            Assert.assertEquals(array1.logicalXor(array2), expected);
            Assert.assertEquals(NDArrays.logicalXor(array1, array2), expected);
            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            array2 = manager.create(new Shape(1, 0, 2));
            expected = manager.create(new Shape(1, 0, 2), DataType.BOOLEAN);
            Assert.assertEquals(array1.logicalXor(array2), expected);
            Assert.assertEquals(NDArrays.logicalXor(array1, array2), expected);
        }
    }

    @Test
    public void testLogicalNot() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {-2f, 0f, 1f});
            NDArray expected = manager.create(new boolean[] {false, true, false});
            Assertions.assertAlmostEquals(array.logicalNot(), expected);
            array = manager.create(new float[] {1f, 2f, -1f, -2f}, new Shape(2, 2));
            expected = manager.create(new boolean[] {false, false, false, false}, new Shape(2, 2));
            Assertions.assertAlmostEquals(array.logicalNot(), expected);
            // test scalar
            array = manager.create(0f);
            expected = manager.create(true);
            Assert.assertEquals(array.logicalNot(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0, 1));
            Assert.assertEquals(array.logicalNot(), array.toType(DataType.BOOLEAN, false));
        }
    }
}
