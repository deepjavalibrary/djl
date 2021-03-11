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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayShapesManipulationOpTest {

    @Test
    public void testSplit() {
        // TODO add more test cases once MXNet split op bug is fixed
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(18f);
            NDList result = array.split(18);
            Assert.assertEquals(result.get(0), manager.create(new float[] {0f}));
            Assert.assertEquals(result.get(8), manager.create(new float[] {8f}));
            Assert.assertEquals(result.get(17), manager.create(new float[] {17f}));

            array = manager.create(new float[] {1f, 2f, 3f, 4f});
            result = array.split(2);
            Assert.assertEquals(result.get(0), manager.create(new float[] {1f, 2f}));
            Assert.assertEquals(result.get(1), manager.create(new float[] {3f, 4f}));
            result = array.split(new long[] {2});
            Assert.assertEquals(result.get(0), manager.create(new float[] {1f, 2f}));
            Assert.assertEquals(result.get(1), manager.create(new float[] {3f, 4f}));

            // special case: indices = empty
            array = manager.arange(6f).reshape(2, 3);
            result = array.split(new long[0]);
            Assert.assertEquals(result.singletonOrThrow(), array);
        }
    }

    @Test
    public void testFlatten() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray expected = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assert.assertEquals(array.flatten(), expected);

            // multi-dim
            array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            expected = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assert.assertEquals(array.flatten(), expected);

            // scalar
            array = manager.create(5f);
            expected = manager.create(new float[] {5f});
            Assert.assertEquals(array.flatten(), expected);

            // zero-dim
            array = manager.create(new Shape(2, 0));
            expected = manager.create(new Shape(0));
            Assert.assertEquals(array.flatten(), expected);
        }
    }

    @Test
    public void testReshape() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f});
            NDArray expected =
                    manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 1, 1, 3));
            Assertions.assertAlmostEquals(array.reshape(2, 1, 1, 3), expected);
            try {
                // only MXNet, PyTorch and TensorFlow support -1
                Assertions.assertAlmostEquals(array.reshape(-1, 1, 1, 3), expected);
            } catch (UnsupportedOperationException ignore) {
                // ignore
            }

            // multi-dim
            array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(3, 2));
            expected = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
            Assertions.assertAlmostEquals(array.reshape(new Shape(2, 3)), expected);
            try {
                // only MXNet, PyTorch and TensorFlow support -1
                Assertions.assertAlmostEquals(array.reshape(new Shape(2, -1)), expected);
            } catch (UnsupportedOperationException ignore) {
                // ignore
            }

            // scalar
            array = manager.create(5f);
            expected = manager.create(new float[] {5f});
            Assertions.assertAlmostEquals(array.reshape(1), expected);
            expected = manager.create(new float[] {5f}, new Shape(1, 1, 1));
            try {
                // only MXNet, PyTorch and TensorFlow support -1
                Assertions.assertAlmostEquals(array.reshape(1, -1, 1), expected);
            } catch (UnsupportedOperationException ignore) {
                // ignore
            }

            // zero-dim
            array = manager.create(new Shape(1, 0));
            expected = manager.create(new Shape(2, 3, 0, 1));
            Assertions.assertAlmostEquals(array.reshape(2, 3, 0, 1), expected);
        }
    }

    @Test
    public void testExpandDim() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f});
            NDArray expected = manager.create(new float[] {1f, 2f}, new Shape(1, 2));
            Assert.assertEquals(array.expandDims(0), expected);

            // multi-dim
            array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            expected = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 1, 2));
            Assert.assertEquals(array.expandDims(1), expected);

            // scalar
            array = manager.create(4f);
            expected = manager.create(new float[] {4f});
            Assert.assertEquals(array.expandDims(0), expected);

            // zero-dim
            array = manager.create(new Shape(2, 1, 0));
            expected = manager.create(new Shape(2, 1, 1, 0));
            Assert.assertEquals(array.expandDims(2), expected);
        }
    }

    @Test
    public void testSqueeze() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.ones(new Shape(1, 2, 1, 3, 1));
            NDArray expected = manager.ones(new Shape(2, 3));
            Assert.assertEquals(array.squeeze(), expected);
            expected = manager.ones(new Shape(1, 2, 3, 1));
            Assert.assertEquals(array.squeeze(2), expected);
            expected = manager.ones(new Shape(2, 1, 3));
            Assert.assertEquals(array.squeeze(new int[] {0, 4}), expected);

            // scalar
            array = manager.create(2f);
            Assert.assertEquals(array.squeeze(), array);
            Assert.assertEquals(array.squeeze(0), array);
            Assert.assertEquals(array.squeeze(new int[] {0}), array);

            // zero-dim
            array = manager.create(new Shape(1, 0, 1, 3, 1));
            expected = manager.create(new Shape(0, 3));
            Assert.assertEquals(array.squeeze(), expected);
            expected = manager.create(new Shape(1, 0, 3, 1));
            Assert.assertEquals(array.squeeze(2), expected);
            expected = manager.create(new Shape(0, 1, 3));
            Assert.assertEquals(array.squeeze(new int[] {0, 4}), expected);
        }
    }

    @Test
    public void testStack() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f, 2f});
            NDArray array2 = manager.create(new float[] {3f, 4f});

            NDArray expected = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            Assert.assertEquals(array1.stack(array2), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array2)), expected);
            expected = manager.create(new float[] {1f, 3f, 2f, 4f}, new Shape(2, 2));
            Assert.assertEquals(array1.stack(array2, 1), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array2), 1), expected);

            array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            array2 = manager.create(new float[] {5f, 6f, 7f, 8f}, new Shape(2, 2));
            expected =
                    manager.create(
                            new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new Shape(2, 2, 2));
            Assert.assertEquals(array1.stack(array2), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array2)), expected);
            expected =
                    manager.create(
                            new float[] {1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f}, new Shape(2, 2, 2));
            Assert.assertEquals(array1.stack(array2, 1), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array2), 1), expected);
            expected =
                    manager.create(
                            new float[] {1f, 5f, 2f, 6f, 3f, 7f, 4f, 8f}, new Shape(2, 2, 2));
            Assert.assertEquals(array1.stack(array2, 2), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array2), 2), expected);

            // scalar
            array1 = manager.create(5f);
            array2 = manager.create(4f);
            expected = manager.create(new float[] {5f, 4f});
            Assert.assertEquals(array1.stack(array2), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array2)), expected);

            // zero-dim
            array1 = manager.create(new Shape(0, 0));
            expected = manager.create(new Shape(2, 0, 0));
            Assert.assertEquals(array1.stack(array1), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array1)), expected);
            expected = manager.create(new Shape(0, 2, 0));
            Assert.assertEquals(array1.stack(array1, 1), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array1), 1), expected);
            expected = manager.create(new Shape(0, 0, 2));
            Assert.assertEquals(array1.stack(array1, 2), expected);
            Assert.assertEquals(NDArrays.stack(new NDList(array1, array1), 2), expected);

            // one array
            array1 = manager.ones(new Shape(2, 2));
            expected = manager.ones(new Shape(1, 2, 2));
            Assert.assertEquals(NDArrays.stack(new NDList(array1)), expected);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testConcat() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new float[] {1f});
            NDArray array2 = manager.create(new float[] {2f});
            NDArray expected = manager.create(new float[] {1f, 2f});
            Assert.assertEquals(NDArrays.concat(new NDList(array1, array2), 0), expected);
            Assert.assertEquals(array1.concat(array2), expected);

            array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            array2 = manager.create(new float[] {5f, 6f, 7f, 8f}, new Shape(2, 2));
            expected = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7, 8f}, new Shape(4, 2));
            Assert.assertEquals(NDArrays.concat(new NDList(array1, array2)), expected);
            Assert.assertEquals(array1.concat(array2), expected);
            expected =
                    manager.create(new float[] {1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f}, new Shape(2, 4));
            Assert.assertEquals(NDArrays.concat(new NDList(array1, array2), 1), expected);
            expected =
                    manager.create(new float[] {1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f}, new Shape(2, 4));
            Assert.assertEquals(NDArrays.concat(new NDList(array1, array2), 1), expected);

            // zero-dim
            array1 = manager.create(new Shape(0, 1));
            expected = manager.create(new Shape(0, 1));
            Assert.assertEquals(array1.concat(array1), expected);
            Assert.assertEquals(NDArrays.concat(new NDList(array1, array1)), expected);
            expected = manager.create(new Shape(0, 2));
            Assert.assertEquals(array1.concat(array1, 1), expected);
            Assert.assertEquals(NDArrays.concat(new NDList(array1, array1), 1), expected);

            // scalar
            array1 = manager.create(1f);
            array2 = manager.create(2f);
            array1.concat(array2);

            // one array
            array1 = manager.ones(new Shape(2, 2));
            expected = manager.ones(new Shape(2, 2));
            Assert.assertEquals(NDArrays.concat(new NDList(array1)), expected);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testConcatNDlist() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(1f);
            NDArray array2 = manager.create(2f);
            NDArrays.concat(new NDList(array1, array2));
        }
    }
}
