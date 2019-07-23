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

import java.util.Arrays;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayShapesManipulationOpTest extends AbstractTest {

    public static void main(String[] args) {
        new NDArrayShapesManipulationOpTest().runTest(args);
    }

    @RunAsTest
    public void testSplit() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDList splitted = original.split(2);
            Assertions.assertEquals(splitted.head(), manager.create(new float[] {1f, 2f}));
            Assertions.assertEquals(splitted.get(1), manager.create(new float[] {3f, 4f}));
            // only test simple case, current numpy split have bug
            splitted = original.split(new int[] {2});
            Assertions.assertEquals(splitted.head(), manager.create(new float[] {1f, 2f}));
            Assertions.assertEquals(splitted.get(1), manager.create(new float[] {3f, 4f}));
        }
    }

    @RunAsTest
    public void testFlatten() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            NDArray flattened = original.flatten();
            NDArray expected = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assertions.assertEquals(flattened, expected);
        }
    }

    @RunAsTest
    public void testReshape() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original =
                    manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(3, 2));
            NDArray reshaped = original.reshape(new Shape(2, 3));
            NDArray expected =
                    manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
            Assertions.assertEquals(reshaped, expected);
            reshaped = original.reshape(new Shape(2, -1));
            Assertions.assertEquals(reshaped, expected);
        }
    }

    @RunAsTest
    public void testExpandDim() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new int[] {1, 2});
            Assertions.assertTrue(
                    Arrays.equals(
                            original.expandDims(0).getShape().getShape(),
                            new Shape(1, 2).getShape()));
        }
    }

    @RunAsTest
    public void testSqueeze() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.ones(new Shape(1, 2, 1, 3, 1));
            Assertions.assertTrue(
                    Arrays.equals(
                            original.squeeze().getShape().getShape(), new Shape(2, 3).getShape()));
            Assertions.assertTrue(
                    Arrays.equals(
                            original.squeeze(2).getShape().getShape(),
                            new Shape(1, 2, 3, 1).getShape()));
            Assertions.assertTrue(
                    Arrays.equals(
                            original.squeeze(new int[] {0, 4}).getShape().getShape(),
                            new Shape(2, 1, 3).getShape()));
        }
    }

    @RunAsTest
    public void testStack() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {1f, 2f});
            NDArray expect = manager.create(new float[] {1f, 2f, 1f, 2f}, new Shape(2, 2));
            Assertions.assertEquals(original.stack(original), expect);
        }
    }

    @RunAsTest
    public void testConcat() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray concatedND = manager.create(new float[] {1f});
            NDArray concatedND2 = manager.create(new float[] {2f});
            NDArray actual = manager.create(new float[] {1f, 2f});

            Assertions.assertEquals(concatedND.concat(new NDArray[] {concatedND2}, 0), actual);
            Assertions.assertEquals(
                    NDArrays.concat(new NDArray[] {concatedND, concatedND2}, 0), actual);
            Assertions.assertEquals(concatedND.concat(concatedND2), actual);
        }
    }
}
