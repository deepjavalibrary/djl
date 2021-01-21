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
}
