/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.integration;

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class PtNDArrayTest {

    @Test
    public void testStringTensor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            String[] str = {"a", "b", "c"};
            NDArray arr = manager.create(str);
            Assert.assertEquals(arr.toString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(true), Arrays.toString(str));

            Assert.assertThrows(UnsupportedOperationException.class, () -> arr.get(0));
        }
    }

    @Test
    public void testLargeTensor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(10 * 2850, 18944), DataType.FLOAT32);
            Assert.assertThrows(EngineException.class, array::toByteArray);
        }
    }

    @Test
    public static void testPtTensorToLongArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // Initialize a 10x10 array with values
            final long[][] data = new long[10][10];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = i * 10 + j;
                }
            }

            // Create an NDArray from the 2D long array
            final NDArray array = manager.create(data);

            // Convert the NDArray to a 1D long array
            final long[] result = array.toLongArray();

            // Assert that the original data matches the data in the NDArray
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    Assert.assertEquals(
                            result[i * data.length + j],
                            data[i][j],
                            "The data in the NDArray does not match the original data.");
                }
            }
        }
    }

    @Test
    public static void testPtTensorToDoubleArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            final double[][] data = new double[10][10];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = i * 10.0 + j;
                }
            }
            final NDArray array = manager.create(data);
            final double[] result = array.toDoubleArray();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    Assert.assertEquals(
                            result[i * data.length + j],
                            data[i][j],
                            0.0001,
                            "The data in the NDArray does not match the original data.");
                }
            }
        }
    }

    @Test
    public static void testPtTensorToFloatArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            final float[][] data = new float[10][10];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = (float) (i * 10.0 + j);
                }
            }
            final NDArray array = manager.create(data);
            final float[] result = array.toFloatArray();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    Assert.assertEquals(
                            result[i * data.length + j],
                            data[i][j],
                            0.0001,
                            "The data in the NDArray does not match the original data.");
                }
            }
        }
    }

    @Test
    public static void testPtTensorToIntArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            final int[][] data = new int[10][10];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = i * 10 + j;
                }
            }
            final NDArray array = manager.create(data);
            final int[] result = array.toIntArray();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    Assert.assertEquals(
                            result[i * data.length + j],
                            data[i][j],
                            "The data in the NDArray does not match the original data.");
                }
            }
        }
    }

    @Test
    public static void testPtTensorToByteArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            final byte[][] data = new byte[10][10];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = (byte) (i * 10 + j);
                }
            }
            final NDArray array = manager.create(data);
            final byte[] result = array.toByteArray();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    Assert.assertEquals(
                            result[i * data.length + j],
                            data[i][j],
                            "The data in the NDArray does not match the original data.");
                }
            }
        }
    }

    @Test
    public static void testPtTensorToBooleanArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            final boolean[][] data = new boolean[10][10];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = (i * 10 + j) % 2 == 0; // Alternately setting true and false
                }
            }
            final NDArray array = manager.create(data);
            final boolean[] result = array.toBooleanArray();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    Assert.assertEquals(
                            result[i * data.length + j],
                            data[i][j],
                            "The data in the NDArray does not match the original data.");
                }
            }
        }
    }
}
