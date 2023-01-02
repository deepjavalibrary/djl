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

import ai.djl.engine.Engine;
import ai.djl.integration.util.TestUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.testing.Assertions;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.FloatBuffer;
import java.util.stream.IntStream;

public class NDArrayCreationOpTest {

    @Test
    public void testCreation() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            // regular case
            NDArray array = manager.create(new float[] {0, 1, 2, 3}, new Shape(2, 2));
            Assert.assertEquals(array.toFloatArray(), new float[] {0, 1, 2, 3});
            Assert.assertEquals(array.getShape(), new Shape(2, 2));
            // test scalar
            array = manager.create(-100f);
            Assert.assertEquals(array.getFloat(), -100f);
            Assert.assertEquals(array.getShape(), new Shape());
            // test zero-dim
            array = manager.create(new float[] {}, new Shape(1, 0));
            Assert.assertEquals(array.getShape(), new Shape(1, 0));
            Assert.assertEquals(array.toArray().length, 0);

            double[] data = IntStream.range(0, 100).mapToDouble(i -> i).toArray();
            array = manager.create(data);
            NDArray expected = manager.arange(0, 100, 1, DataType.FLOAT64, array.getDevice());
            // test 1d
            Assert.assertEquals(array, expected);
            // test 2d
            double[][] data2D = {data, data};
            array = manager.create(data2D);
            expected = NDArrays.stack(new NDList(manager.create(data), manager.create(data)));
            Assert.assertEquals(array, expected);

            // test boolean
            array = manager.create(new boolean[] {true, false, true, false}, new Shape(2, 2));
            expected = manager.create(new int[] {1, 0, 1, 0}, new Shape(2, 2));
            Assert.assertEquals(array.toType(DataType.INT32, false), expected);
            Assert.assertEquals(array, expected.toType(DataType.BOOLEAN, false));
        }
    }

    @Test
    public void testCreateCSRMatrix() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            float[] expected = {7, 8, 9};
            FloatBuffer buf = FloatBuffer.wrap(expected);
            long[] indptr = {0, 2, 2, 3};
            long[] indices = {0, 2, 1};
            NDArray nd = manager.createCSR(buf, indptr, indices, new Shape(3, 4));
            float[] array = nd.toDense().toFloatArray();
            Assert.assertEquals(array[0], expected[0]);
            Assert.assertEquals(array[2], expected[1]);
            Assert.assertEquals(array[9], expected[2]);
            Assert.assertTrue(nd.isSparse());
        }
    }

    @Test
    public void testCreateRowSparseMatrix() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            float[] expected = {1, 2, 3, 4, 5, 6};
            FloatBuffer buf = FloatBuffer.wrap(expected);
            long[] indices = {0, 1, 3};
            NDArray nd = manager.createRowSparse(buf, new Shape(3, 2), indices, new Shape(4, 2));
            float[] array = nd.toDense().toFloatArray();
            Assert.assertEquals(array[0], expected[0]);
            Assert.assertEquals(array[1], expected[1]);
            Assert.assertEquals(array[2], expected[2]);
            Assert.assertEquals(array[3], expected[3]);
            Assert.assertEquals(array[6], expected[4]);
            Assert.assertEquals(array[7], expected[5]);
            Assert.assertTrue(nd.isSparse());
        }
    }

    @Test
    public void testCreateCooMatrix() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            long[][] indices = {{0, 1, 1}, {2, 0, 2}};
            float[] values = {3, 4, 5};
            FloatBuffer buf = FloatBuffer.wrap(values);
            NDArray nd = manager.createCoo(buf, indices, new Shape(2, 4));
            NDArray expected =
                    manager.create(new float[] {0, 0, 3, 0, 4, 0, 5, 0}, new Shape(2, 4));
            Assert.assertTrue(nd.isSparse());
            Assert.assertEquals(nd.toDense(), expected);
        }
    }

    @Test
    public void testCreateNDArrayAndConvertToSparse() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray nd = manager.ones(new Shape(3, 5));
            try {
                // Only MXNet support CSR
                NDArray sparse = nd.toSparse(SparseFormat.CSR);
                Assert.assertSame(sparse.getSparseFormat(), SparseFormat.CSR);
            } catch (UnsupportedOperationException ignore) {
                // ignore
            }
            try {
                // Only PyTorch support COO
                NDArray sparse = nd.toSparse(SparseFormat.COO);
                Assert.assertSame(sparse.getSparseFormat(), SparseFormat.COO);
            } catch (UnsupportedOperationException ignore) {
                // ignore
            }
        }
    }

    @Test
    public void testDuplicate() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.zeros(new Shape(5));
            NDArray expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f});
            NDArray duplicate = array.duplicate();
            Assert.assertEquals(duplicate, expected);
            Assert.assertNotSame(duplicate, expected);

            // test multi-dim
            array = manager.ones(new Shape(2, 3));
            expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f}, new Shape(2, 3));
            Assert.assertEquals(array, expected);
            duplicate = array.duplicate();
            Assert.assertEquals(duplicate, expected);
            Assert.assertNotSame(duplicate, expected);

            // test scalar
            array = manager.zeros(new Shape());
            expected = manager.create(0f);
            duplicate = array.duplicate();
            Assert.assertEquals(duplicate, expected);
            Assert.assertNotSame(duplicate, expected);

            // test zero-dim
            array = manager.zeros(new Shape(0, 1));
            expected = manager.create(new Shape(0, 1));
            duplicate = array.duplicate();
            Assert.assertEquals(duplicate, expected);
            Assert.assertNotSame(duplicate, expected);
        }
    }

    @Test
    public void testZeros() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.zeros(new Shape(5));
            NDArray expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f});
            Assertions.assertAlmostEquals(array, expected);

            // test multi-dim
            array = manager.zeros(new Shape(2, 3));
            expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 0f}, new Shape(2, 3));
            Assertions.assertAlmostEquals(array, expected);

            // test scalar
            array = manager.zeros(new Shape());
            expected = manager.create(0f);
            Assertions.assertAlmostEquals(array, expected);

            // test zero-dim
            array = manager.zeros(new Shape(0, 1));
            expected = manager.create(new Shape(0, 1));
            Assertions.assertAlmostEquals(array, expected);
        }
    }

    @Test
    public void testOnes() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.ones(new Shape(5));
            NDArray expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f});
            Assertions.assertAlmostEquals(array, expected);

            // test multi-dim
            array = manager.ones(new Shape(2, 3));
            expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f}, new Shape(2, 3));
            Assertions.assertAlmostEquals(array, expected);

            // test scalar
            array = manager.ones(new Shape());
            expected = manager.create(1f);
            Assertions.assertAlmostEquals(array, expected);

            // test zero-dim
            array = manager.ones(new Shape(0, 1));
            expected = manager.create(new Shape(0, 1));
            Assertions.assertAlmostEquals(array, expected);
        }
    }

    @Test
    public void testFull() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.full(new Shape(5), 3);
            NDArray expected = manager.create(new int[] {3, 3, 3, 3, 3});
            Assert.assertEquals(array, expected);
            array = manager.full(new Shape(6), 5f);
            expected = manager.create(new float[] {5f, 5f, 5f, 5f, 5f, 5f});
            Assert.assertEquals(array, expected);

            // test multi-dim
            array = manager.full(new Shape(2, 3), -100);
            expected =
                    manager.create(new int[] {-100, -100, -100, -100, -100, -100}, new Shape(2, 3));
            Assert.assertEquals(array, expected);
            array = manager.full(new Shape(3, 2), 4f);
            expected = manager.create(new float[] {4f, 4f, 4f, 4f, 4f, 4f}, new Shape(3, 2));
            Assert.assertEquals(array, expected);

            // test scalar
            array = manager.full(new Shape(), 1f);
            expected = manager.create(1f);
            Assert.assertEquals(array, expected);
            array = manager.full(new Shape(), 0);
            expected = manager.create(0);
            Assert.assertEquals(array, expected);

            // test zero-dim
            array = manager.ones(new Shape(0, 1));
            expected = manager.create(new Shape(0, 1));
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testZerosLike() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.create(new Shape(5));
            NDArray expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f});
            Assert.assertEquals(array.zerosLike(), expected);

            // test multi-dim
            array = manager.create(new Shape(2, 3));
            expected = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 0f}, new Shape(2, 3));
            Assert.assertEquals(array.zerosLike(), expected);

            // test scalar
            array = manager.create(new Shape());
            expected = manager.create(0f);
            Assert.assertEquals(array.zerosLike(), expected);

            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array.zerosLike(), array);
        }
    }

    @Test
    public void testOnesLike() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.create(new Shape(5));
            NDArray expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f});
            Assert.assertEquals(array.onesLike(), expected);

            // test multi-dim
            array = manager.create(new Shape(2, 3));
            expected = manager.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f}, new Shape(2, 3));
            Assert.assertEquals(array.onesLike(), expected);

            // test scalar
            array = manager.create(new Shape());
            expected = manager.create(1f);
            Assert.assertEquals(array.onesLike(), expected);

            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array.onesLike(), array);
        }
    }

    @Test
    public void testArange() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.arange(0, 10, 1);
            NDArray expected = manager.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
            Assert.assertEquals(array, expected);
            array = manager.arange(0, 10, 1);
            Assert.assertEquals(array, expected);
            array = manager.arange(10);
            Assert.assertEquals(array, expected);
            array = manager.arange(3.5f);
            expected = manager.create(new float[] {0f, 1f, 2f, 3f});
            Assert.assertEquals(array, expected);
            array = manager.arange(0.1f, 5.4f, 0.3f);
            expected =
                    manager.create(
                            new float[] {
                                0.1f, 0.4f, 0.7f, 1f, 1.3f, 1.6f, 1.9f, 2.2f, 2.5f, 2.8f, 3.1f,
                                3.4f, 3.7f, 4f, 4.3f, 4.6f, 4.9f, 5.2f
                            });
            Assertions.assertAlmostEquals(array, expected);
            array = manager.arange(0.0f, 2.0f, 0.3f);
            expected = manager.create(new float[] {0f, 0.3f, 0.6f, 0.9f, 1.2f, 1.5f, 1.8f});
            Assertions.assertAlmostEquals(array, expected);

            // test 0 dimension
            array = manager.arange(10.0f, 0.0f, 1.0f);
            expected = manager.create(new Shape(0));
            Assert.assertEquals(array, expected);
            array = manager.arange(0.0f, -2.0f);
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testEye() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray expected = manager.eye(2);
            NDArray array = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
            Assert.assertEquals(array, expected);
            array = manager.eye(2, 3, 0);
            expected = manager.create(new float[] {1f, 0f, 0f, 0f, 1f, 0f}, new Shape(2, 3));
            Assert.assertEquals(array, expected);
            array = manager.eye(3, 4, 0);
            expected =
                    manager.create(
                            new float[] {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f},
                            new Shape(3, 4));
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testLinspace() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray array = manager.linspace(0.0f, 9.0f, 10, true, manager.getDevice());
            NDArray expected = manager.arange(10.0f);
            Assert.assertEquals(array, expected);
            array = manager.linspace(0.0f, 10.0f, 10, false, manager.getDevice());
            Assert.assertEquals(array, expected);
            array = manager.linspace(10, 0, 10, false, manager.getDevice());
            expected = manager.create(new float[] {10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f});
            Assert.assertEquals(array, expected);
            array = manager.linspace(10, 10, 10);
            expected = manager.ones(new Shape(10)).mul(10);
            Assert.assertEquals(array, expected);

            // test corner case
            array = manager.linspace(0, 10, 0);
            expected = manager.create(new Shape(0));
            Assert.assertEquals(array, expected);
        }
    }

    @Test
    public void testRandomInteger() {
        PairList<Long, Long> testCases = new PairList<>();
        testCases.add(0L, 2L);
        testCases.add(1000000L, 2000000L);
        testCases.add(-1234567L, 1234567L);
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            for (Pair<Long, Long> testCase : testCases) {
                long low = testCase.getKey();
                long high = testCase.getValue();
                NDArray randLong =
                        manager.randomInteger(low, high, new Shape(100, 100), DataType.INT64);
                double mean = randLong.toType(DataType.FLOAT64, false).mean().getDouble();
                long max = randLong.max().getLong();
                long min = randLong.min().getLong();
                Assert.assertTrue(max < high);
                Assert.assertTrue(min >= low);
                Assert.assertTrue(mean >= low && mean < high);
            }
        }
    }

    @Test
    public void testRandomPermutation() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            long size = 3;
            NDArray array = manager.randomPermutation(size);
            Assert.assertTrue(
                    array.contentEquals(manager.create(new long[] {0, 1, 2}))
                            || array.contentEquals(manager.create(new long[] {0, 2, 1}))
                            || array.contentEquals(manager.create(new long[] {1, 0, 2}))
                            || array.contentEquals(manager.create(new long[] {1, 2, 0}))
                            || array.contentEquals(manager.create(new long[] {2, 0, 1}))
                            || array.contentEquals(manager.create(new long[] {2, 1, 0})));
        }
    }

    @Test
    public void testRandomUniform() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray uniform = manager.randomUniform(0, 10, new Shape(1000, 1000));
            Assert.assertTrue(uniform.min().getFloat() >= 0f);
            Assert.assertTrue(uniform.max().getFloat() < 10f);
            Assertions.assertAlmostEquals(uniform.mean().getFloat(), 5f, 1e-2f, 1e-2f);
        }
    }

    @Test
    public void testRandomNormal() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray normal = manager.randomNormal(new Shape(1000, 1000));
            NDArray mean = normal.mean();
            NDArray std = normal.sub(mean).pow(2).mean();
            Assertions.assertAlmostEquals(mean.getFloat(), 0f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(std.getFloat(), 1f, 2e-2f, 2e-2f);
        }
    }

    @Test
    public void testTruncatedNormal() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray normal = manager.truncatedNormal(new Shape(1000, 1000));
            Assertions.assertAlmostEquals(normal.mean().getFloat(), 0f, 2e-2f, 2e-2f);
            Assert.assertTrue(normal.gte(-2).all().getBoolean());
            Assert.assertTrue(normal.lte(2).all().getBoolean());
        }
    }

    @Test
    public void testFixedSeed() {
        // TensorFlow fixed random seed requires restart process.
        TestUtils.requiresEngine("MXNet", "PyTorch");

        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            int fixedSeed = 1234;
            Engine engine = Engine.getEngine(TestUtils.getEngine());
            engine.setRandomSeed(fixedSeed);
            NDArray expectedUniform = manager.randomUniform(-10, 10, new Shape(10, 10));
            engine.setRandomSeed(fixedSeed);
            NDArray actualUniform = manager.randomUniform(-10, 10, new Shape(10, 10));
            Assertions.assertAlmostEquals(expectedUniform, actualUniform, 1e-2f, 1e-2f);

            engine.setRandomSeed(fixedSeed);
            NDArray expectedNormal = manager.randomNormal(new Shape(100, 100));
            engine.setRandomSeed(fixedSeed);
            NDArray actualNormal = manager.randomNormal(new Shape(100, 100));
            Assertions.assertAlmostEquals(expectedNormal, actualNormal, 1e-2f, 1e-2f);
        }
    }

    @Test
    public void testSamplePoisson() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray lam = manager.create(new double[] {1., 8.5});

            NDArray poisson = manager.samplePoisson(lam);
            Assert.assertEquals(poisson.getShape(), new Shape(2));

            poisson = manager.samplePoisson(lam, new Shape(1000, 1000));
            Assert.assertEquals(poisson.getShape(), new Shape(2, 1000, 1000));

            NDArray mean = poisson.mean(new int[] {1, 2});
            NDArray std = poisson.transpose(1, 2, 0).sub(mean).pow(2).mean(new int[] {0, 1});

            Assertions.assertAlmostEquals(mean.toFloatArray()[0], 1f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(mean.toFloatArray()[1], 8.5f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(std.toFloatArray()[0], 1f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(std.toFloatArray()[1], 8.5f, 2e-2f, 2e-2f);
        }
    }

    @Test
    public void testSampleGamma() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray alpha = manager.create(new double[] {0., 2.5});
            NDArray beta = manager.create(new double[] {1., 0.7});

            NDArray gamma = manager.sampleGamma(alpha, beta);
            Assert.assertEquals(gamma.getShape(), new Shape(2));

            gamma = manager.sampleGamma(alpha, beta, new Shape(1000, 1000));
            Assert.assertEquals(gamma.getShape(), new Shape(2, 1000, 1000));

            NDArray mean = gamma.mean(new int[] {1, 2});
            NDArray std = gamma.transpose(1, 2, 0).sub(mean).pow(2).mean(new int[] {0, 1});

            Assertions.assertAlmostEquals(mean.toFloatArray()[0], 0f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(mean.toFloatArray()[1], 1.75f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(std.toFloatArray()[0], 0f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(std.toFloatArray()[1], 1.225f, 2e-2f, 2e-2f);
        }
    }

    @Test
    public void testSampleNormal() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray mu = manager.create(new double[] {0., 2.5});
            NDArray sigma = manager.create(new double[] {1., 3.7});
            NDArray normal = manager.sampleNormal(mu, sigma);
            Assert.assertEquals(normal.getShape(), new Shape(2));

            normal = manager.sampleNormal(mu, sigma, new Shape(1000, 1000));
            Assert.assertEquals(normal.getShape(), new Shape(2, 1000, 1000));

            NDArray mean = normal.mean(new int[] {1, 2});
            NDArray std = normal.transpose(1, 2, 0).sub(mean).pow(2).mean(new int[] {0, 1});

            Assertions.assertAlmostEquals(mean.toFloatArray()[0], 0f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(mean.toFloatArray()[1], 2.5f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(std.toFloatArray()[0], 1f, 2e-2f, 2e-2f);
            Assertions.assertAlmostEquals(std.toFloatArray()[1], 13.69f, 2e-2f, 2e-2f);
        }
    }

    @Test
    public void testHanningWindow() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDArray result = manager.hanningWindow(6);
            NDArray expected = manager.create(new float[] {0.0f, 0.25f, 0.75f, 1.0f, 0.75f, 0.25f});
            Assertions.assertAlmostEquals(result, expected);
        }
    }
}
