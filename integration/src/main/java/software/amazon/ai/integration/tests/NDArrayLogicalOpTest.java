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

import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayLogicalOpTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", NDArrayLogicalOpTest.class.getName()};
        new IntegrationTest().runTests(cmd);
    }

    @RunAsTest
    public void testLogicalAnd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original1 = manager.arange(10);
            NDArray original2 = manager.arange(10);
            NDArray actual = manager.create(new float[] {0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
            Assertions.assertEquals(original1.logicalAnd(original2), actual);
            Assertions.assertEquals(NDArrays.logicalAnd(original1, original2), actual);
        }
    }

    @RunAsTest
    public void testLogicalOr() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original1 = manager.arange(10);
            NDArray original2 = manager.arange(10);
            NDArray actual = manager.create(new float[] {0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
            Assertions.assertEquals(original1.logicalOr(original2), actual);
            Assertions.assertEquals(NDArrays.logicalOr(original1, original2), actual);
        }
    }

    @RunAsTest
    public void testLogicalXor() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original1 = manager.arange(10);
            NDArray original2 = manager.arange(10);
            NDArray actual = manager.zeros(new Shape(10));
            Assertions.assertEquals(original1.logicalXor(original2), actual);
            Assertions.assertEquals(NDArrays.logicalXor(original1, original2), actual);
        }
    }

    @RunAsTest
    public void testLogicalNot() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new double[] {-2., 0., 1.});
            NDArray expect = manager.create(new double[] {0.0, 1.0, 0.0});
            Assertions.assertAlmostEquals(original.logicalNot(), expect);
        }
    }
}
