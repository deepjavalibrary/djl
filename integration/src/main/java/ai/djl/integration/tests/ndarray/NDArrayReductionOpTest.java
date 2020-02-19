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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayReductionOpTest {

    @Test
    public void testMax() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 5f, 1f});
            Assert.assertEquals(array.max().getFloat(), 5f);

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float maxAll = array.max().getFloat();
            Assert.assertEquals(maxAll, 8f, "Incorrect max all");

            NDArray maxAxes = array.max(new int[] {1});
            NDArray expected = manager.create(new float[] {4f, 8f});
            Assert.assertEquals(maxAxes, expected, "Incorrect max axes");

            NDArray maxKeep = array.max(new int[] {0}, true);
            expected = manager.create(new float[] {6f, 8f}, new Shape(1, 2));
            Assert.assertEquals(maxKeep, expected, "Incorrect max keep");

            // test scalar
            array = manager.create(5f);
            Assert.assertEquals(array.max().getFloat(), 5f);

            // TODO MXNet engine crash
            // zero-dim
            // array = manager.create(new Shape(1, 0));
            // array.max();
        }
    }

    @Test
    public void testMin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {2f, 1f, 5f, 0f});
            Assert.assertEquals(array.min().getFloat(), 0f);

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float minAll = array.min().getFloat();
            Assert.assertEquals(minAll, 2f, "Incorrect min all");

            NDArray minAxes = array.min(new int[] {1});
            NDArray expected = manager.create(new float[] {2f, 6f});
            Assert.assertEquals(minAxes, expected, "Incorrect min axes");

            NDArray minKeep = array.min(new int[] {0}, true);
            expected = manager.create(new float[] {2f, 4f}, new Shape(1, 2));
            Assert.assertEquals(minKeep, expected, "Incorrect min keep");

            // test scalar
            array = manager.create(0f);
            Assert.assertEquals(array.min().getFloat(), 0f);

            // TODO MXNet engine crash
            // zero-dim
            // array = manager.create(new Shape(0, 2, 0));
            // array.min();
        }
    }

    @Test
    public void testSum() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 5f});
            Assert.assertEquals(array.sum().getFloat(), 11f);

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float sumAll = array.sum().getFloat();
            Assert.assertEquals(sumAll, 20f, "Incorrect sum all");
            NDArray sumAxes = array.sum(new int[] {1});
            NDArray expected = manager.create(new float[] {6f, 14f});
            Assert.assertEquals(sumAxes, expected, "Incorrect sum axes");

            NDArray sumKeep = array.sum(new int[] {0}, true);
            expected = manager.create(new float[] {8f, 12f}, new Shape(1, 2));
            Assert.assertEquals(sumKeep, expected, "Incorrect sum keep");

            // scalar
            array = manager.create(2f);
            Assert.assertEquals(array.sum().getFloat(), 2f);
            // TODO wait for MXNet numpy sum bug fix
            // zero-dim
            // array = manager.create(new Shape(1, 1, 0));
            // Assert.assertEquals(array.sum().getFloat(), 0f);
        }
    }

    @Test
    public void testProd() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assert.assertEquals(array.prod().getFloat(), 24f);

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));

            float prodAll = array.prod().getFloat();
            Assert.assertEquals(prodAll, 384f, "Incorrect prod axes");

            NDArray prodAxes = array.prod(new int[] {1});
            NDArray expected = manager.create(new float[] {8f, 48f});
            Assert.assertEquals(prodAxes, expected, "Incorrect prod axes");

            NDArray prodKeep = array.prod(new int[] {0}, true);
            expected = manager.create(new float[] {12f, 32f}, new Shape(1, 2));
            Assert.assertEquals(prodKeep, expected, "Incorrect prod keep");

            // scalar
            array = manager.create(5f);
            Assert.assertEquals(array.prod().getFloat(), 5f);
            // TODO wait for MXNet numpy prod bug fix
            // zero-dim
            // array = manager.create(new Shape(0, 0, 0));
            // Assert.assertEquals(array.prod().getFloat(), 1f);
        }
    }

    @Test
    public void testMean() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assert.assertEquals(array.mean().getFloat(), 2.5f);

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float meanAll = array.mean().getFloat();
            Assert.assertEquals(meanAll, 5f, "Incorrect mean all");
            NDArray meanAxes = array.mean(new int[] {1});
            NDArray expected = manager.create(new float[] {3f, 7f});
            Assert.assertEquals(meanAxes, expected, "Incorrect mean axes");

            NDArray meanKeep = array.mean(new int[] {0}, true);
            expected = manager.create(new float[] {4f, 6f}, new Shape(1, 2));
            Assert.assertEquals(meanKeep, expected, "Incorrect mean keep");

            // scalar
            array = manager.create(5f);
            Assert.assertEquals(array.mean().getFloat(), 5f);
            // TODO disable for now until MXNet np mean bug fix
            // zero-dim
            // array = manager.create(new Shape(0, 0, 0));
            // Assert.assertEquals(array.mean().getFloat(), Float.NaN);
        }
    }

    @Test
    public void testTrace() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(8.0f).reshape(new Shape(2, 2, 2)).trace();
            NDArray expect = manager.create(new float[] {6f, 8f});
            Assert.assertEquals(original, expect);
            original = manager.arange(24.0f).reshape(new Shape(2, 2, 2, 3)).trace();
            Assert.assertEquals(original.getShape(), new Shape(2, 3));
        }
    }
}
