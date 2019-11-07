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

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayReductionOpTest {

    @Test(expectedExceptions = EngineException.class)
    public void testMax() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 5f, 1f});
            Assert.assertEquals(5f, array.max().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float maxAll = array.max().getFloat();
            Assert.assertEquals(8f, maxAll, "Incorrect max all");

            NDArray maxAxes = array.max(new int[] {1});
            NDArray maxAxesActual = manager.create(new float[] {4f, 8f});
            Assert.assertEquals(maxAxesActual, maxAxes, "Incorrect max axes");

            NDArray maxKeep = array.max(new int[] {0}, true);
            NDArray maxKeepActual = manager.create(new float[] {6f, 8f}, new Shape(1, 2));
            Assert.assertEquals(maxKeepActual, maxKeep, "Incorrect max keep");

            // test scalar
            array = manager.create(5f);
            Assert.assertEquals(5f, array.max().getFloat());

            // TODO MXNet engine crash
            // zero-dim
            // array = manager.create(new Shape(1, 0));
            // array.max();
        }
    }

    @Test(expectedExceptions = EngineException.class)
    public void testMin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {2f, 1f, 5f, 0f});
            Assert.assertEquals(0f, array.min().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float minAll = array.min().getFloat();
            Assert.assertEquals(2f, minAll, "Incorrect min all");

            NDArray minAxes = array.min(new int[] {1});
            NDArray minAxesActual = manager.create(new float[] {2f, 6f});
            Assert.assertEquals(minAxesActual, minAxes, "Incorrect min axes");

            NDArray minKeep = array.min(new int[] {0}, true);
            NDArray minKeepActual = manager.create(new float[] {2f, 4f}, new Shape(1, 2));
            Assert.assertEquals(minKeepActual, minKeep, "Incorrect min keep");

            // test scalar
            array = manager.create(0f);
            Assert.assertEquals(0f, array.min().getFloat());

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
            Assert.assertEquals(11f, array.sum().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float sumAll = array.sum().getFloat();
            Assert.assertEquals(20f, sumAll, "Incorrect sum all");
            NDArray sumAxes = array.sum(new int[] {1});
            NDArray sumAxesActual = manager.create(new float[] {6f, 14f});
            Assert.assertEquals(sumAxesActual, sumAxes, "Incorrect sum axes");

            NDArray sumKeep = array.sum(new int[] {0}, true);
            NDArray sumKeepActual = manager.create(new float[] {8f, 12f}, new Shape(1, 2));
            Assert.assertEquals(sumKeepActual, sumKeep, "Incorrect sum keep");

            // scalar
            array = manager.create(2f);
            Assert.assertEquals(2f, array.sum().getFloat());
            // TODO wait for MXNet numpy sum bug fix
            // zero-dim
            // array = manager.create(new Shape(1, 1, 0));
            // Assert.assertEquals(0f, array.sum().getFloat());
        }
    }

    @Test
    public void testProd() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assert.assertEquals(24f, array.prod().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));

            float prodAll = array.prod().getFloat();
            Assert.assertEquals(384f, prodAll, "Incorrect prod axes");

            NDArray prodAxes = array.prod(new int[] {1});
            NDArray prodAxesActual = manager.create(new float[] {8f, 48f});
            Assert.assertEquals(prodAxesActual, prodAxes, "Incorrect prod axes");

            NDArray prodKeep = array.prod(new int[] {0}, true);
            NDArray prodKeepActual = manager.create(new float[] {12f, 32f}, new Shape(1, 2));
            Assert.assertEquals(prodKeepActual, prodKeep, "Incorrect prod keep");

            // scalar
            array = manager.create(5f);
            Assert.assertEquals(5f, array.prod().getFloat());
            // TODO wait for MXNet numpy prod bug fix
            // zero-dim
            // array = manager.create(new Shape(0, 0, 0));
            // Assert.assertEquals(1f, array.prod().getFloat());
        }
    }

    @Test
    public void testMean() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assert.assertEquals(2.5f, array.mean().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float meanAll = array.mean().getFloat();
            Assert.assertEquals(5f, meanAll, "Incorrect mean all");
            NDArray meanAxes = array.mean(new int[] {1});
            NDArray meanAxesActual = manager.create(new float[] {3f, 7f});
            Assert.assertEquals(meanAxesActual, meanAxes, "Incorrect mean axes");

            NDArray meanKeep = array.mean(new int[] {0}, true);
            NDArray meanKeepActual = manager.create(new float[] {4f, 6f}, new Shape(1, 2));
            Assert.assertEquals(meanKeepActual, meanKeep, "Incorrect mean keep");

            // scalar
            array = manager.create(5f);
            Assert.assertEquals(5f, array.mean().getFloat());
            // TODO disable for now until MXNet np mean bug fix
            // zero-dim
            // array = manager.create(new Shape(0, 0, 0));
            // Assert.assertEquals(Float.NaN, array.mean().getFloat());
        }
    }

    @Test
    public void testTrace() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(8).reshape(new Shape(2, 2, 2)).trace();
            NDArray expect = manager.create(new float[] {6f, 8f});
            Assert.assertEquals(original, expect);
            original = manager.arange(24).reshape(new Shape(2, 2, 2, 3)).trace();
            Assert.assertEquals(new Shape(2, 3), original.getShape());
        }
    }
}
