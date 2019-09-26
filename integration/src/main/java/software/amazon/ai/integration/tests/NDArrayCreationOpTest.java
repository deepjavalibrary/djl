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

import java.nio.FloatBuffer;
import java.util.stream.IntStream;
import org.testng.annotations.Test;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;

public class NDArrayCreationOpTest {

    @Test
    public void testCreation() throws FailedTestException {
        // TODO add more test cases to make it robust
        try (NDManager manager = NDManager.newBaseManager()) {
            // test scalar
            NDArray array = manager.create(-100f);
            Assertions.assertEquals(-100f, array.getFloat());
            Assertions.assertEquals(new Shape(), array.getShape());

            // test zero-dim
            array = manager.create(new float[] {}, new Shape(1, 0));
            Assertions.assertEquals(new Shape(1, 0), array.getShape());
            Assertions.assertEquals(0, array.toArray().length);

            double[] data = IntStream.range(0, 100).mapToDouble(i -> i).toArray();
            array = manager.create(data);
            NDArray actual = manager.arange(0, 100, 1, DataType.FLOAT64, array.getDevice());
            // test 1d
            Assertions.assertEquals(actual, array);
            // test 2d
            double[][] data2D = {data, data};
            array = manager.create(data2D);
            actual = NDArrays.stack(new NDArray[] {manager.create(data), manager.create(data)});
            Assertions.assertEquals(actual, array);
        }
    }

    @Test
    public void testCreateCSRMatrix() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            float[] actual = {7, 8, 9};
            FloatBuffer buf = FloatBuffer.wrap(actual);
            long[] indptr = {0, 2, 2, 3};
            long[] indices = {0, 2, 1};
            NDArray nd = factory.createCSR(buf, indptr, indices, new Shape(3, 4));
            float[] array = nd.toFloatArray();
            Assertions.assertEquals(actual[0], array[0]);
            Assertions.assertEquals(actual[1], array[2]);
            Assertions.assertEquals(actual[2], array[9]);
            Assertions.assertTrue(nd.isSparse());
        }
    }

    @Test
    public void testCreateRowSparseMatrix() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            float[] actual = {1, 2, 3, 4, 5, 6};
            FloatBuffer buf = FloatBuffer.wrap(actual);
            long[] indices = {0, 1, 3};
            NDArray nd = factory.createRowSparse(buf, new Shape(3, 2), indices, new Shape(4, 2));
            float[] array = nd.toFloatArray();
            Assertions.assertEquals(actual[0], array[0]);
            Assertions.assertEquals(actual[1], array[1]);
            Assertions.assertEquals(actual[2], array[2]);
            Assertions.assertEquals(actual[3], array[3]);
            Assertions.assertEquals(actual[4], array[6]);
            Assertions.assertEquals(actual[5], array[7]);
            Assertions.assertTrue(nd.isSparse());
        }
    }

    @Test
    public void testCreateNDArrayAndConvertToSparse() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            NDArray nd = factory.ones(new Shape(3, 5));
            NDArray sparse = nd.toSparse(SparseFormat.CSR);
            Assertions.assertTrue(sparse.getSparseFormat() == SparseFormat.CSR);
        }
    }

    @Test
    public void testArange() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
            NDArray actual = manager.arange(0, 10, 1);
            Assertions.assertEquals(actual, array);
            actual = manager.arange(0, 10, 1);
            Assertions.assertEquals(actual, array);
            actual = manager.arange(10);
            Assertions.assertEquals(actual, array);
            // test 0 dimension
            array = manager.arange(10, 0, 1);
            actual = manager.create(new Shape(0));
            Assertions.assertEquals(actual, array);
            array = manager.arange(0, -2);
            Assertions.assertEquals(actual, array);
        }
    }

    // TODO disable for now
    @Test(enabled = false)
    public void testEye() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.eye(2);
            NDArray actual = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            Assertions.assertEquals(actual, array);
            array = manager.eye(2, 3, 0);
            actual = manager.create(new float[] {1f, 0f, 0f, 0f, 1f, 0f}, new Shape(2, 3));
            Assertions.assertEquals(actual, array);
            array = manager.eye(3, 4, 0);
            actual =
                    manager.create(
                            new float[] {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f},
                            new Shape(3, 4));
            Assertions.assertEquals(actual, array);
        }
    }

    @Test
    public void testLinspace() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.linspace(0.0, 9.0, 10, true, manager.getDevice());
            NDArray actual = manager.arange(10);
            Assertions.assertEquals(actual, array);
            array = manager.linspace(0.0, 10.0, 10, false, manager.getDevice());
            Assertions.assertEquals(actual, array);
            array = manager.linspace(10, 0, 10, false, manager.getDevice());
            actual = manager.create(new float[] {10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f});
            Assertions.assertEquals(actual, array);
            array = manager.linspace(10, 10, 10);
            actual = manager.ones(new Shape(10)).mul(10);
            Assertions.assertEquals(actual, array);

            // test corner case
            array = manager.linspace(0, 10, 0);
            actual = manager.create(new Shape(0));
            Assertions.assertEquals(actual, array);
        }
    }
}
