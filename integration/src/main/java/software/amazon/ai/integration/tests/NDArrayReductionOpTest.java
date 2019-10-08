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

import org.testng.annotations.Test;
import software.amazon.ai.engine.EngineException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayReductionOpTest {

    @Test
    public void testMax() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 5f, 1f});
            Assertions.assertEquals(5f, array.max().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float maxAll = array.max().getFloat();
            Assertions.assertEquals(8f, maxAll, "Incorrect max all");

            NDArray maxAxes = array.max(new int[] {1});
            NDArray maxAxesActual = manager.create(new float[] {4f, 8f});
            Assertions.assertEquals(maxAxesActual, maxAxes, "Incorrect max axes");

            NDArray maxKeep = array.max(new int[] {0}, true);
            NDArray maxKeepActual = manager.create(new float[] {6f, 8f}, new Shape(1, 2));
            Assertions.assertEquals(maxKeepActual, maxKeep, "Incorrect max keep");

            // test scalar
            array = manager.create(5f);
            Assertions.assertEquals(5f, array.max().getFloat());
            // zero-dim
            array = manager.create(new Shape(1, 0));
            Assertions.assertThrows(array::max, EngineException.class);
        }
    }

    @Test
    public void testMin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {2f, 1f, 5f, 0f});
            Assertions.assertEquals(0f, array.min().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float minAll = array.min().getFloat();
            Assertions.assertEquals(2f, minAll, "Incorrect min all");

            NDArray minAxes = array.min(new int[] {1});
            NDArray minAxesActual = manager.create(new float[] {2f, 6f});
            Assertions.assertEquals(minAxesActual, minAxes, "Incorrect min axes");

            NDArray minKeep = array.min(new int[] {0}, true);
            NDArray minKeepActual = manager.create(new float[] {2f, 4f}, new Shape(1, 2));
            Assertions.assertEquals(minKeepActual, minKeep, "Incorrect min keep");

            // test scalar
            array = manager.create(0f);
            Assertions.assertEquals(0f, array.min().getFloat());
            // zero-dim
            array = manager.create(new Shape(0, 2, 0));
            Assertions.assertThrows(array::min, EngineException.class);
        }
    }

    @Test
    public void testSum() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 5f});
            Assertions.assertEquals(11f, array.sum().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float sumAll = array.sum().getFloat();
            Assertions.assertEquals(20f, sumAll, "Incorrect sum all");
            NDArray sumAxes = array.sum(new int[] {1});
            NDArray sumAxesActual = manager.create(new float[] {6f, 14f});
            Assertions.assertEquals(sumAxesActual, sumAxes, "Incorrect sum axes");

            NDArray sumKeep = array.sum(new int[] {0}, true);
            NDArray sumKeepActual = manager.create(new float[] {8f, 12f}, new Shape(1, 2));
            Assertions.assertEquals(sumKeepActual, sumKeep, "Incorrect sum keep");

            // scalar
            array = manager.create(2f);
            Assertions.assertEquals(2f, array.sum().getFloat());
            // zero-dim
            array = manager.create(new Shape(1, 1, 0));
            Assertions.assertEquals(0f, array.sum().getFloat());
        }
    }

    @Test
    public void testProd() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assertions.assertEquals(24f, array.prod().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));

            float prodAll = array.prod().getFloat();
            Assertions.assertEquals(384f, prodAll, "Incorrect prod axes");

            NDArray prodAxes = array.prod(new int[] {1});
            NDArray prodAxesActual = manager.create(new float[] {8f, 48f});
            Assertions.assertEquals(prodAxesActual, prodAxes, "Incorrect prod axes");

            NDArray prodKeep = array.prod(new int[] {0}, true);
            NDArray prodKeepActual = manager.create(new float[] {12f, 32f}, new Shape(1, 2));
            Assertions.assertEquals(prodKeepActual, prodKeep, "Incorrect prod keep");

            // scalar
            array = manager.create(5f);
            Assertions.assertEquals(5f, array.prod().getFloat());
            // TODO wait for MXNet numpy prod bug fix
            // zero-dim
            // array = manager.create(new Shape(0, 0, 0));
            // Assertions.assertEquals(1f, array.prod().getFloat());
        }
    }

    @Test
    public void testMean() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f});
            Assertions.assertEquals(2.5f, array.mean().getFloat());

            array = manager.create(new float[] {2f, 4f, 6f, 8f}, new Shape(2, 2));
            float meanAll = array.mean().getFloat();
            Assertions.assertEquals(5f, meanAll, "Incorrect mean all");
            NDArray meanAxes = array.mean(new int[] {1});
            NDArray meanAxesActual = manager.create(new float[] {3f, 7f});
            Assertions.assertEquals(meanAxesActual, meanAxes, "Incorrect mean axes");

            NDArray meanKeep = array.mean(new int[] {0}, true);
            NDArray meanKeepActaul = manager.create(new float[] {4f, 6f}, new Shape(1, 2));
            Assertions.assertEquals(meanKeepActaul, meanKeep, "Incorrect mean keep");

            // scalar
            array = manager.create(5f);
            Assertions.assertEquals(5f, array.mean().getFloat());
            // TODO disable for now until MXNet np mean bug fix
            // zero-dim
            // array = manager.create(new Shape(0, 0, 0));
            // Assertions.assertEquals(Float.NaN, array.mean().getFloat());
        }
    }
    // TODO update libmxnet to get trace op
    @Test(enabled = false)
    public void testTrace() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.arange(8).reshape(new Shape(2, 2, 2)).trace();
            NDArray expect = manager.create(new float[] {6f, 8f});
            Assertions.assertEquals(original, expect);
            original = manager.arange(24).reshape(new Shape(2, 2, 2, 3)).trace();
            Assertions.assertTrue(original.getShape().equals(new Shape(2, 3)));
        }
    }
}
