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
import java.util.Arrays;
import java.util.stream.Stream;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;

public class NDArrayCreationOpTest {

    public static void main(String[] args) {
        String[] cmd = {"-c", NDArrayCreationOpTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testCreateCSRMatrix() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            float[] input = {7, 8, 9};
            FloatBuffer buf = FloatBuffer.wrap(input);
            long[] indptr = {0, 2, 2, 3};
            long[] indices = {0, 2, 1};
            NDArray nd = factory.createCSR(buf, indptr, indices, new Shape(3, 4));
            float[] array = nd.toFloatArray();
            Assertions.assertTrue(input[0] == array[0]);
            Assertions.assertTrue(input[1] == array[2]);
            Assertions.assertTrue(input[2] == array[9]);
            Assertions.assertTrue(nd.isSparse());
        }
    }

    @RunAsTest
    public void testCreateRowSparseMatrix() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            float[] input = {1, 2, 3, 4, 5, 6};
            FloatBuffer buf = FloatBuffer.wrap(input);
            long[] indices = {0, 1, 3};
            NDArray nd = factory.createRowSparse(buf, new Shape(3, 2), indices, new Shape(4, 2));
            float[] array = nd.toFloatArray();
            Assertions.assertTrue(input[0] == array[0]);
            Assertions.assertTrue(input[1] == array[1]);
            Assertions.assertTrue(input[2] == array[2]);
            Assertions.assertTrue(input[3] == array[3]);
            Assertions.assertTrue(input[4] == array[6]);
            Assertions.assertTrue(input[5] == array[7]);
            Assertions.assertTrue(nd.isSparse());
        }
    }

    @RunAsTest
    public void testCreateNDArrayAndConvertToSparse() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            NDArray nd = factory.ones(new Shape(3, 5));
            NDArray sparse = nd.toSparse(SparseFormat.CSR);
            Assertions.assertTrue(sparse.getSparseFormat() == SparseFormat.CSR);
        }
    }

    @RunAsTest
    public void testArange() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray expectedND =
                    manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
            NDArray testedND = manager.arange(0, 10, 1);
            Assertions.assertEquals(testedND, expectedND);
            testedND = manager.arange(0, 10, 1);
            Assertions.assertEquals(testedND, expectedND);
            testedND = manager.arange(10);
            Assertions.assertEquals(testedND, expectedND);
        }
    }

    @RunAsTest
    public void testEye() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.eye(2);
            NDArray expect = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            Assertions.assertEquals(original, expect);
            original = manager.eye(2, 3, 0);
            expect = manager.create(new float[] {1f, 0f, 0f, 0f, 1f, 0f}, new Shape(2, 3));
            Assertions.assertEquals(original, expect);
            original = manager.eye(3, 4, 0);
            expect =
                    manager.create(
                            new float[] {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f},
                            new Shape(3, 4));
            Assertions.assertEquals(original, expect);
        }
    }

    @RunAsTest
    public void testLinspace() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray expectedND =
                    manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
            NDArray testedND = manager.linspace(0.0, 9.0, 10, true, null);
            Assertions.assertEquals(testedND, expectedND);
        }
    }
}
