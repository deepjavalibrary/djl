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

import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayReductionOpTest extends AbstractTest {

    public static void main(String[] args) {
        new NDArrayReductionOpTest().runTest(args);
    }

    @RunAsTest
    public void testAmax() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {2, 4, 6, 8}, new Shape(2, 2));

            Float maxAll = (Float) original.max();
            Assertions.assertEquals(8, maxAll, "Incorrect max all");

            NDArray maxAxes = original.max(new int[] {1});
            NDArray maxAxesExpected = manager.create(new float[] {4, 8});
            Assertions.assertEquals(maxAxesExpected, maxAxes, "Incorrect max axes");

            NDArray maxKeep = original.max(new int[] {0}, true);
            NDArray maxKeepExpected = manager.create(new float[] {6, 8}, new Shape(1, 2));
            Assertions.assertEquals(maxKeepExpected, maxKeep, "Incorrect max keep");
        }
    }

    @RunAsTest
    public void testAmin() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {2, 4, 6, 8}, new Shape(2, 2));

            Float minAll = (Float) original.min();
            Assertions.assertEquals(2, minAll, "Incorrect min all");

            NDArray minAxes = original.min(new int[] {1});
            NDArray minAxesExpected = manager.create(new float[] {2, 6});
            Assertions.assertEquals(minAxesExpected, minAxes, "Incorrect min axes");

            NDArray minKeep = original.min(new int[] {0}, true);
            NDArray minKeepExpected = manager.create(new float[] {2, 4}, new Shape(1, 2));
            Assertions.assertEquals(minKeepExpected, minKeep, "Incorrect min keep");
        }
    }

    @RunAsTest
    public void testSum() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {2, 4, 6, 8}, new Shape(2, 2));

            Float sumAll = (Float) original.sum();
            Assertions.assertEquals(20, sumAll, "Incorrect sum all");
            NDArray sumAxes = original.sum(new int[] {1});
            NDArray sumAxesExpected = manager.create(new float[] {6, 14});
            Assertions.assertEquals(sumAxesExpected, sumAxes, "Incorrect sum axes");

            NDArray sumKeep = original.sum(new int[] {0}, true);
            NDArray sumKeepExpected = manager.create(new float[] {8, 12}, new Shape(1, 2));
            Assertions.assertEquals(sumKeepExpected, sumKeep, "Incorrect sum keep");
        }
    }

    @RunAsTest
    public void testProd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {2, 4, 6, 8}, new Shape(2, 2));

            Float prodAll = (Float) original.prod();
            Assertions.assertEquals(384, prodAll, "Incorrect max axes");
            if (prodAll != 384) {
                throw new FailedTestException("Incorrect prod all");
            }

            NDArray prodAxes = original.prod(new int[] {1});
            NDArray prodAxesExpected = manager.create(new float[] {8, 48});
            Assertions.assertEquals(prodAxesExpected, prodAxes, "Incorrect prod axes");

            NDArray prodKeep = original.prod(new int[] {0}, true);
            NDArray prodKeepExpected = manager.create(new float[] {12, 32}, new Shape(1, 2));
            Assertions.assertEquals(prodKeepExpected, prodKeep, "Incorrect prod keep");
        }
    }

    @RunAsTest
    public void testMean() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {2, 4, 6, 8}, new Shape(2, 2));

            Float meanAll = (Float) original.mean();
            Assertions.assertEquals(5, meanAll, "Incorrect mean all");
            NDArray meanAxes = original.mean(new int[] {1});
            NDArray meanAxesExpected = manager.create(new float[] {3, 7});
            Assertions.assertEquals(meanAxesExpected, meanAxes, "Incorrect mean axes");

            NDArray meanKeep = original.mean(new int[] {0}, true);
            NDArray meanKeepExpected = manager.create(new float[] {4, 6}, new Shape(1, 2));
            Assertions.assertEquals(meanKeepExpected, meanKeep, "Incorrect mean keep");
        }
    }

    @RunAsTest
    public void testTrace() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(8).reshape(new Shape(2, 2, 2)).trace();
            NDArray expect = manager.create(new float[] {6f, 8f});
            Assertions.assertEquals(original, expect);
            original = manager.arange(24).reshape(new Shape(2, 2, 2, 3)).trace();
            Assertions.assertTrue(original.getShape().equals(new Shape(2, 3)));
        }
    }
}
