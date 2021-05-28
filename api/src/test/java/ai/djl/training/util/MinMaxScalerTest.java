/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.util;

import static org.testng.Assert.assertTrue;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.testng.annotations.Test;

/**
 * UnitTest for MinMaxScaler.
 *
 * @author erik.bamberg@web.de
 */
public class MinMaxScalerTest {

    /**
     * data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
     *
     * <p>>>> scaler = MinMaxScaler() >>> print(scaler.fit(data)) MinMaxScaler() >>>
     * print(scaler.data_max_) [ 1. 18.] >>> print(scaler.transform(data)) [[0. 0. ] [0.25 0.25]
     * [0.5 0.5 ] [1. 1. ]] >>> print(scaler.transform([[2, 2]])) [[1.5 0. ]]
     */
    private static final float[][] TESTDATA = {{-1f, 2f}, {-0.5f, 6f}, {0f, 10f}, {1f, 18f}};

    private static final float[] EXPECTED_MIN = {-1f, 2f};
    private static final float[] EXPECTED_MAX = {1, 18};
    private static final float[][] EXPECTED_TRANSFORMED_DATA = {
        {0f, 0f}, {0.25f, 0.25f}, {0.5f, 0.5f}, {1f, 1f}
    };
    private static final float[][] EXPECTED_SECOND_TRANSFORMED_DATA = {{1.5f, 0f}};

    @Test
    public void testMinMaxWithDefaultAxis() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(TESTDATA);
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(data);
            assertTrue(manager.create(EXPECTED_MIN).contentEquals(scaler.getMin()));
            assertTrue(manager.create(EXPECTED_MAX).contentEquals(scaler.getMax()));
        }
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testMinOnUnfittedScalerThrowsException() {
        MinMaxScaler scaler = new MinMaxScaler();
        scaler.getMin();
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testMaxOnUnfittedScalerThrowsException() {
        MinMaxScaler scaler = new MinMaxScaler();
        scaler.getMax();
    }

    @Test
    public void testFitReturnSelf() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(TESTDATA);
            MinMaxScaler scaler = new MinMaxScaler();
            MinMaxScaler returned = scaler.fit(data);
            assertTrue(scaler == returned);
        }
    }

    @Test
    public void testTransform() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(TESTDATA);
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(data);
            NDArray transformed = scaler.transform(data);
            assertTrue(manager.create(EXPECTED_TRANSFORMED_DATA).contentEquals(transformed));
            // now test other testdata fitted to the same MinMax
            NDArray data2 = manager.create(new float[][] {{2f, 2f}});
            NDArray transformed2 = scaler.transform(data2);
            assertTrue(
                    manager.create(EXPECTED_SECOND_TRANSFORMED_DATA).contentEquals(transformed2));
        }
    }

    @Test
    public void testTransformSimpleArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[][] {{0f, 4f}, {2f, 2f}, {2f, 3f}});

            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(data);
            NDArray transformed = scaler.transform(data);
            assertTrue(
                    manager.create(new float[][] {{0f, 1f}, {1f, 0f}, {1f, 0.5f}})
                            .contentEquals(transformed));
        }
    }

    @Test
    public void testTransformSimpleArrayWithRange() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[][] {{0f, 4f}, {2f, 2f}, {2f, 3f}});

            MinMaxScaler scaler = new MinMaxScaler();
            scaler.optRange(5f, 10f);
            scaler.fit(data);
            NDArray transformed = scaler.transform(data);
            assertTrue(
                    manager.create(new float[][] {{5f, 10f}, {10f, 5f}, {10f, 7.5f}})
                            .contentEquals(transformed));
        }
    }

    @Test
    public void testInverseTransformSimpleArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[][] {{0f, 4f}, {2f, 2f}, {2f, 3f}});

            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(data);
            NDArray iData = manager.create(new float[][] {{0f, 1f}, {1f, 0f}, {1f, 0.5f}});
            NDArray transformed = scaler.inverseTransform(iData);
            assertTrue(data.contentEquals(transformed));
            // assertion is that calculation are not done in-place, but returns a new array
            assertTrue(iData != transformed);
        }
    }

    @Test
    public void testInverseTransformSimpleArrayWithRange() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[][] {{0f, 4f}, {2f, 2f}, {2f, 3f}});

            MinMaxScaler scaler = new MinMaxScaler();
            scaler.optRange(5f, 10f);
            scaler.fit(data);
            NDArray iData = manager.create(new float[][] {{5f, 10f}, {10f, 5f}, {10f, 7.5f}});
            NDArray transformed = scaler.inverseTransform(iData);
            assertTrue(data.contentEquals(transformed));
            // assertion is that calculation are not done in-place, but returns a new array
            assertTrue(iData != transformed);
        }
    }

    @Test
    public void testInverseTransformInplaceSimpleArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[][] {{0f, 4f}, {2f, 2f}, {2f, 3f}});

            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(data);
            NDArray iData = manager.create(new float[][] {{0f, 1f}, {1f, 0f}, {1f, 0.5f}});
            NDArray transformed = scaler.inverseTransformi(iData);
            assertTrue(data.contentEquals(transformed));
            // assertion is that calculation are done in-place
            assertTrue(iData == transformed);
        }
    }

    @Test
    public void testTransformWithRange() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[][] {{0f, 4f}, {2f, 2f}, {2f, 3f}});

            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(data);
            scaler.optRange(0f, 2f);
            NDArray transformed = scaler.transform(data);
            assertTrue(
                    manager.create(new float[][] {{0f, 2f}, {2f, 0f}, {2f, 1f}})
                            .contentEquals(transformed));
        }
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testInverseTransformWithUnfittedScaler() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[] {0f});
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.inverseTransform(data);
        }
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testInverseInPlaceTransformWithUnfittedScaler() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(new float[] {0f});
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.inverseTransformi(data);
        }
    }
}
