/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Tests for usages of the {@link NDIndex} including get/set. */
public class NDIndexTest {

    @Test
    public void testEmptyIndex() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            Assert.assertEquals(original.get(new NDIndex()), original);
        }
    }

    @Test
    public void testFixedNegativeIndex() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(4));
            NDArray expected = manager.create(4f);
            NDArray actual = original.get("-1");
            Assert.assertEquals(actual, expected);
        }
    }

    @Test
    public void testPick() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {1f, 4f}, new Shape(2, 1));
            NDArray actual =
                    original.get(
                            new NDIndex().addAllDim().addPickDim(manager.create(new int[] {0, 1})));
            Assert.assertEquals(actual, expected);

            // The difference between take and pick used combined with addAllDim()
            NDArray yHat = manager.create(new float[][] {{0.1f, 0.3f, 0.6f}, {0.3f, 0.2f, 0.5f}});
            NDArray yGet = yHat.get(new NDIndex(":, {}", manager.create(new int[] {0, 2})));
            NDArray yPick =
                    yHat.get(
                            new NDIndex().addAllDim().addPickDim(manager.create(new int[] {0, 2})));
            Assert.assertEquals(yGet, manager.create(new float[][] {{0.1f, 0.6f}, {0.3f, 0.5f}}));
            Assert.assertEquals(yPick, manager.create(new float[][] {{0.1f}, {0.5f}}));
        }
    }

    @Test
    public void testGather() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(20f).reshape(-1, 4);
            NDArray index = manager.create(new float[] {0, 0, 2, 1, 1, 2}, new Shape(3, 2));
            NDArray actual = original.gather(index, 1);
            NDArray expected = manager.create(new float[] {0, 0, 6, 5, 9, 10}, new Shape(3, 2));
            Assert.assertEquals(actual, expected);
        }
    }

    @Test
    public void testTake() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(1, 7f).reshape(-1, 3);
            NDArray index = manager.create(new float[] {0, 4, 1, 2}, new Shape(2, 2));
            NDArray actual = original.take(index);
            NDArray expected = manager.create(new float[] {1, 5, 2, 3}, new Shape(2, 2));
            Assert.assertEquals(actual, expected);
        }
    }

    @Test
    public void testGet() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            Assert.assertEquals(original.get(new NDIndex()), original);

            NDArray getAt = original.get(0);
            NDArray expected = manager.create(new float[] {1f, 2f});
            Assert.assertEquals(getAt, expected);

            Assert.assertEquals(original.get("0,:"), expected);
            Assert.assertEquals(original.get("0,*"), expected);

            NDArray getSlice = original.get("1:");
            expected = manager.create(new float[] {3f, 4f}, new Shape(1, 2));
            Assert.assertEquals(getSlice, expected);

            NDArray getStepSlice = original.get("1::2");
            Assert.assertEquals(getStepSlice, expected);

            original = manager.arange(120).reshape(2, 3, 4, 5);
            NDArray getEllipsis = original.get("0,2, ...  ");
            expected = manager.arange(40, 60).reshape(4, 5);
            Assert.assertEquals(getEllipsis, expected);

            getEllipsis = original.get("...,0:2,2");
            expected =
                    manager.create(new int[] {2, 7, 22, 27, 42, 47, 62, 67, 82, 87, 102, 107})
                            .reshape(2, 3, 2);
            Assert.assertEquals(getEllipsis, expected);

            getEllipsis = original.get("1,...,2,3:5:2");
            expected = manager.create(new int[] {73, 93, 113}).reshape(3, 1);
            Assert.assertEquals(getEllipsis, expected);

            getEllipsis = original.get("...");
            Assert.assertEquals(getEllipsis, original);

            // get from boolean array
            original = manager.arange(10).reshape(2, 5);
            NDArray bool = manager.create(new boolean[] {true, false});
            expected = manager.arange(5).reshape(1, 5);
            Assert.assertEquals(original.get(bool), expected);

            // get from integer array (higher rank included) or float array
            original = manager.arange(1, 7f).reshape(-1, 2);
            NDArray index = manager.create(new long[] {0, 0, 1, 2}, new Shape(2, 2));
            NDArray indexFloat = manager.create(new float[] {0, 0, 1, 2}, new Shape(2, 2));
            NDArray actual = original.get(index);
            NDArray actual2 = original.get(indexFloat);
            expected = manager.create(new float[] {1, 2, 1, 2, 3, 4, 5, 6}, new Shape(2, 2, 2));
            Assert.assertEquals(actual, expected);
            Assert.assertEquals(actual2, expected);

            // indexing with boolean, slice, and integer array (higher rank included) or float array
            original = manager.arange(3 * 3 * 3 * 3).reshape(3, 3, 3, 3);
            NDArray bool1 = manager.create(new boolean[] {true, false, true});
            NDArray index1 = manager.create(new long[] {2, 2}, new Shape(1, 2));
            NDArray index2 = manager.create(new float[] {0, 1}, new Shape(1, 2));
            actual = original.get(":{}, {}, {}, {}", 2, index1, bool1, index2);
            expected = manager.create(new int[] {18, 25, 45, 52}, new Shape(2, 1, 2));
            Assert.assertEquals(actual, expected);

            // indexing with null, slice and integer array (higher rank included) or float array
            original = manager.arange(3 * 3 * 3).reshape(3, 3, 3);
            index1 = manager.create(new float[] {0, 1}, new Shape(2));
            index2 = manager.create(new long[] {0, 0, 2, 1}, new Shape(2, 2));
            actual = original.get(":{}, {}, {}, {}", 2, index1, index2, null);
            expected = manager.create(new int[] {0, 3, 2, 4, 9, 12, 11, 13}, new Shape(2, 2, 2, 1));
            Assert.assertEquals(actual, expected);
        }
    }

    @Test
    public void testEmptyArrayClosing() {
        // This is to check the resource closing issue in MXNet engine is circumvented.
        // MXNetError: Check failed: delay_alloc:
        try (NDManager manager = NDManager.newBaseManager()) {
            manager.create(new Shape(2, 2)).get(":4, 3:4");
            manager.create(new Shape(2)).get("3:3");
            manager.create(new Shape(2)).get("2:3");
        }
    }

    @Test
    public void testSetArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {9, 10, 3, 4}, new Shape(2, 2));
            NDArray value = manager.create(new float[] {9, 10});
            original.set(new NDIndex(0), value);
            Assert.assertEquals(original, expected);
            original = manager.arange(0, 8).reshape(2, 4);
            expected = manager.create(new int[] {0, 1, 9, 10, 4, 5, 11, 12}, new Shape(2, 4));
            original.set(new NDIndex(":, 2:"), manager.arange(9, 13).reshape(2, 2));
            Assert.assertEquals(original, expected);

            // set by index array
            original = manager.arange(1, 10).reshape(3, 3);
            NDArray index = manager.create(new float[] {0, 1}, new Shape(2));
            value = manager.create(new int[] {666, 777, 888, 999}, new Shape(2, 2));
            original.set(new NDIndex("{}, :{}", index, 2), value);
            expected =
                    manager.create(new int[] {666, 777, 3, 888, 999, 6, 7, 8, 9}, new Shape(3, 3));
            Assert.assertEquals(original, expected);
        }
    }

    @Test
    public void testSetArrayBroadcast() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2, 1));
            NDArray expected = manager.create(new float[] {9, 9, 3, 4}, new Shape(2, 2, 1));
            NDArray value = manager.create(new float[] {9});
            original.set(new NDIndex(0), value);
            Assert.assertEquals(original, expected);
        }
    }

    @Test
    public void testSetNumber() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {9, 9, 3, 4}, new Shape(2, 2));
            original.set(new NDIndex(0), 9);
            Assert.assertEquals(original, expected);

            original = manager.arange(4f).reshape(2, 2);
            expected = manager.ones(new Shape(2, 2));
            original.set(new NDIndex("..."), 1);
            Assert.assertEquals(original, expected);

            original = manager.arange(4f).reshape(2, 2);
            expected = manager.create(new float[] {1, 1, 1, 3}).reshape(2, 2);
            original.set(new NDIndex("..., 0"), 1);
            Assert.assertEquals(original, expected);

            // set by index array
            original = manager.arange(1, 10).reshape(3, 3);
            NDArray index = manager.create(new long[] {0, 1}, new Shape(2));
            original.set(new NDIndex("{}, :{}", index, 2), 666);
            expected =
                    manager.create(new int[] {666, 666, 3, 666, 666, 6, 7, 8, 9}, new Shape(3, 3));
            Assert.assertEquals(original, expected);

            original = manager.arange(1, 10).reshape(3, 3);
            original.set(index, 666);
            expected =
                    manager.create(
                            new int[] {666, 666, 666, 666, 666, 666, 7, 8, 9}, new Shape(3, 3));
            Assert.assertEquals(original, expected);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testSetScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {0, 2, 3, 4}, new Shape(2, 2));
            original.setScalar(new NDIndex(0, 0), 0);
            Assert.assertEquals(original, expected);
            original.setScalar(new NDIndex(0), 1);
        }
    }

    @Test
    public void testSetByFunction() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(1, 10).reshape(3, 3);
            NDArray expected = manager.create(new int[] {4, 10, 16});
            NDIndex index = new NDIndex(":, 1");
            original.set(index, nd -> nd.mul(2));
            Assert.assertEquals(original.get(index), expected);

            original = manager.arange(6).reshape(3, 2);
            expected = manager.create(new int[] {6, 8, 10});
            index = new NDIndex("... , 1");
            original.set(index, nd -> nd.add(5));
            Assert.assertEquals(original.get(index), expected);
        }
    }

    @Test
    public void testSetByFunctionIncrements() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.ones(new Shape(1, 5));
            original.set(new NDIndex(":, 0::2"), array -> array.mul(-1).add(1));
            NDArray expected = manager.create(new float[][] {{0, 1, 0, 1, 0}});
            Assert.assertEquals(original, expected);
        }
    }

    @Test
    public void testPut() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {1, 8, 666, 77}, new Shape(2, 2));
            NDArray idx = manager.create(new float[] {2, 3, 1}, new Shape(3));
            NDArray data = manager.create(new float[] {666, 77, 8}, new Shape(3));
            Assert.assertEquals(original.put(idx, data), expected);
        }
    }
}
