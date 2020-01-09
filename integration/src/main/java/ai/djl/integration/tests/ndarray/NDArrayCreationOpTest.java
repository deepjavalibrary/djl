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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.FloatBuffer;
import java.util.stream.IntStream;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayCreationOpTest {

    @Test
    public void testCreation() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test scalar
            NDArray array = manager.create(-100f);
            Assert.assertEquals(array.getFloat(), -100f);
            Assert.assertEquals(array.getShape(), new Shape());

            // test zero-dim
            array = manager.create(new float[] {}, new Shape(1, 0));
            Assert.assertEquals(array.getShape(), new Shape(1, 0));
            Assert.assertEquals(array.toArray().length, 0);

            double[] data = IntStream.range(0, 100).mapToDouble(i -> i).toArray();
            array = manager.create(data);
            NDArray expected = manager.arange(0, 100, 1, DataType.FLOAT64, array.getDevice());
            // test 1d
            Assert.assertEquals(array, expected);
            // test 2d
            double[][] data2D = {data, data};
            array = manager.create(data2D);
            expected = NDArrays.stack(new NDList(manager.create(data), manager.create(data)));
            Assert.assertEquals(array, expected);

            // test boolean
            array = manager.create(new boolean[] {true, false, true, false}, new Shape(2, 2));
            expected = manager.create(new int[] {1, 0, 1, 0}, new Shape(2, 2));
            Assert.assertEquals(array.toType(DataType.INT32, false), expected);
            Assert.assertEquals(array, expected.toType(DataType.BOOLEAN, false));
        }
    }

    @Test
    public void testCreateCSRMatrix() {
        try (NDManager manager = NDManager.newBaseManager()) {
            float[] expected = {7, 8, 9};
            FloatBuffer buf = FloatBuffer.wrap(expected);
            long[] indptr = {0, 2, 2, 3};
            long[] indices = {0, 2, 1};
            NDArray nd = manager.createCSR(buf, indptr, indices, new Shape(3, 4));
            float[] array = nd.toFloatArray();
            Assert.assertEquals(array[0], expected[0]);
            Assert.assertEquals(array[2], expected[1]);
            Assert.assertEquals(array[9], expected[2]);
            Assert.assertTrue(nd.isSparse());
        }
    }

    @Test
    public void testCreateRowSparseMatrix() {
        try (NDManager manager = NDManager.newBaseManager()) {
            float[] expected = {1, 2, 3, 4, 5, 6};
            FloatBuffer buf = FloatBuffer.wrap(expected);
            long[] indices = {0, 1, 3};
            NDArray nd = manager.createRowSparse(buf, new Shape(3, 2), indices, new Shape(4, 2));
            float[] array = nd.toFloatArray();
            Assert.assertEquals(array[0], expected[0]);
            Assert.assertEquals(array[1], expected[1]);
            Assert.assertEquals(array[2], expected[2]);
            Assert.assertEquals(array[3], expected[3]);
            Assert.assertEquals(array[6], expected[4]);
            Assert.assertEquals(array[7], expected[5]);
            Assert.assertTrue(nd.isSparse());
        }
    }

    @Test
    public void testCreateNDArrayAndConvertToSparse() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray nd = manager.ones(new Shape(3, 5));
            NDArray sparse = nd.toSparse(SparseFormat.CSR);
            Assert.assertSame(sparse.getSparseFormat(), SparseFormat.CSR);
        }
    }

    @Test
    public void testZeros() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(5));
            NDArray expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f});
            Assert.assertEquals(array, expected);

            // test multi-dim
            array = manager.zeros(new Shape(2, 3));
            expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 0f}, new Shape(2, 3));
            Assert.assertEquals(array, expected);

            // test scalar
            array = manager.zeros(new Shape());
            expected = manager.create(0f);
            Assert.assertEquals(array, expected);

            // test zero-dim
            array = manager.zeros(new Shape(0, 1));
            expected = manager.create(new Shape(0, 1));
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testOnes() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.ones(new Shape(5));
            NDArray expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f});
            Assert.assertEquals(array, expected);

            // test multi-dim
            array = manager.ones(new Shape(2, 3));
            expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f}, new Shape(2, 3));
            Assert.assertEquals(array, expected);

            // test scalar
            array = manager.ones(new Shape());
            expected = manager.create(1f);
            Assert.assertEquals(array, expected);

            // test zero-dim
            array = manager.ones(new Shape(0, 1));
            expected = manager.create(new Shape(0, 1));
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testZerosLike() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new Shape(5));
            NDArray expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f});
            Assert.assertEquals(array.zerosLike(), expected);

            // test multi-dim
            array = manager.create(new Shape(2, 3));
            expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 0f}, new Shape(2, 3));
            Assert.assertEquals(array.zerosLike(), expected);

            // test scalar
            array = manager.create(new Shape());
            expected = manager.create(0f);
            Assert.assertEquals(array.zerosLike(), expected);

            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array.zerosLike(), array);
        }
    }

    @Test
    public void testOnesLike() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new Shape(5));
            NDArray expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f});
            Assert.assertEquals(array.onesLike(), expected);

            // test multi-dim
            array = manager.create(new Shape(2, 3));
            expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f}, new Shape(2, 3));
            Assert.assertEquals(array.onesLike(), expected);

            // test scalar
            array = manager.create(new Shape());
            expected = manager.create(1f);
            Assert.assertEquals(array.onesLike(), expected);

            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array.onesLike(), array);
        }
    }

    @Test
    public void testArange() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(0, 10, 1);
            NDArray expected = manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
            Assert.assertEquals(array, expected);
            array = manager.arange(0, 10, 1);
            Assert.assertEquals(array, expected);
            array = manager.arange(10);
            Assert.assertEquals(array, expected);
            array = manager.arange(3.5);
            expected = manager.create(new float[] {0f, 1f, 2f, 3f});
            Assert.assertEquals(array, expected);
            array = manager.arange(0.1, 5.4, 0.3);
            expected =
                    manager.create(
                            new float[] {
                                0.1f, 0.4f, 0.7f, 1f, 1.3f, 1.6f, 1.9f, 2.2f, 2.5f, 2.8f, 3.1f,
                                3.4f, 3.7f, 4f, 4.3f, 4.6f, 4.9f, 5.2f
                            });
            Assertions.assertAlmostEquals(array, expected);
            array = manager.arange(0, 2, 0.3);
            expected = manager.create(new float[] {0f, 0.3f, 0.6f, 0.9f, 1.2f, 1.5f, 1.8f});
            Assertions.assertAlmostEquals(array, expected);

            // test 0 dimension
            array = manager.arange(10, 0, 1);
            expected = manager.create(new Shape(0));
            Assert.assertEquals(array, expected);
            array = manager.arange(0, -2);
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testEye() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray expected = manager.eye(2);
            NDArray array = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            Assert.assertEquals(array, expected);
            array = manager.eye(2, 3, 0);
            expected = manager.create(new float[] {1f, 0f, 0f, 0f, 1f, 0f}, new Shape(2, 3));
            Assert.assertEquals(array, expected);
            array = manager.eye(3, 4, 0);
            expected =
                    manager.create(
                            new float[] {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f},
                            new Shape(3, 4));
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testLinspace() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.linspace(0.0, 9.0, 10, true, manager.getDevice());
            NDArray expected = manager.arange(10);
            Assert.assertEquals(array, expected);
            array = manager.linspace(0.0, 10.0, 10, false, manager.getDevice());
            Assert.assertEquals(array, expected);
            array = manager.linspace(10, 0, 10, false, manager.getDevice());
            expected = manager.create(new float[] {10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f});
            Assert.assertEquals(array, expected);
            array = manager.linspace(10, 10, 10);
            expected = manager.ones(new Shape(10)).mul(10);
            Assert.assertEquals(array, expected);

            // test corner case
            array = manager.linspace(0, 10, 0);
            expected = manager.create(new Shape(0));
            Assert.assertEquals(array, expected);
        }
    }
}
