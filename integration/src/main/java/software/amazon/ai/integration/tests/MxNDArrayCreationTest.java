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
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;

public class MxNDArrayCreationTest extends AbstractTest {

    public static void main(String[] args) {
        new MxNDArrayCreationTest().runTest(args);
    }

    @RunAsTest
    public void testCreateCSRMatrix() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            float[] input = new float[] {7, 8, 9};
            FloatBuffer buf = FloatBuffer.wrap(input);
            long[] indptr = new long[] {0, 2, 2, 3};
            long[] indices = new long[] {0, 2, 1};
            NDArray nd = factory.createCSR(new Shape(3, 4), buf, indptr, indices);
            float[] array = nd.toFloatArray();
            Assertions.assertStatement(input[0] == array[0]);
            Assertions.assertStatement(input[1] == array[2]);
            Assertions.assertStatement(input[2] == array[9]);
            Assertions.assertStatement(nd.isSparse());
        }
    }

    @RunAsTest
    public void testCreateRowSparseMatrix() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            float[] input = new float[] {1, 2, 3, 4, 5, 6};
            FloatBuffer buf = FloatBuffer.wrap(input);
            long[] indices = new long[] {0, 1, 3};
            NDArray nd = factory.createRowSparse(new Shape(4, 2), buf, new Shape(3, 2), indices);
            float[] array = nd.toFloatArray();
            Assertions.assertStatement(input[0] == array[0]);
            Assertions.assertStatement(input[1] == array[1]);
            Assertions.assertStatement(input[2] == array[2]);
            Assertions.assertStatement(input[3] == array[3]);
            Assertions.assertStatement(input[4] == array[6]);
            Assertions.assertStatement(input[5] == array[7]);
            Assertions.assertStatement(nd.isSparse());
        }
    }

    @RunAsTest
    public void testCreateNDArrayAndConvertToSparse() throws FailedTestException {
        try (NDManager factory = NDManager.newBaseManager()) {
            NDArray nd = factory.ones(new Shape(3, 5));
            NDArray sparse = nd.toSparse(SparseFormat.CSR);
            Assertions.assertStatement(sparse.getSparseFormat() == SparseFormat.CSR);
        }
    }
}
