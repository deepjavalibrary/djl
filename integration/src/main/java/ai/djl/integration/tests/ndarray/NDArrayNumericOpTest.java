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
import ai.djl.testing.Assertions;
import java.util.stream.DoubleStream;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayNumericOpTest {

    @Test
    public void testNegation() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {6, 9, -12, -11, 0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(x -> -x).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.neg(), expected);
            Assertions.assertInPlaceEquals(array.negi(), expected, array);
            // test multi-dim
            data = new double[] {-2.2, 2.2, 3, -0.2, 2.76, 0.0002};
            array = manager.create(data, new Shape(2, 3));
            data = DoubleStream.of(data).map(x -> -x).toArray();
            expected = manager.create(data, new Shape(2, 3));
            Assert.assertEquals(array.neg(), expected);
            Assertions.assertInPlaceEquals(array.negi(), expected, array);
            // test scalar
            array = manager.create(3f);
            expected = manager.create(-3f);
            Assert.assertEquals(array.neg(), expected);
            Assertions.assertInPlaceEquals(array.negi(), expected, array);
            // test zero-dim
            array = manager.create(new Shape(2, 0, 1));
            expected = manager.create(new Shape(2, 0, 1));
            Assert.assertEquals(array.neg(), expected);
            Assertions.assertInPlaceEquals(array.negi(), expected, array);
        }
    }

    @Test
    public void testAbs() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::abs).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.abs(), expected);
            // test multi-dim
            data = new double[] {1.2, 98.34, 2.34, -0.456, 2, -22};
            array = manager.create(data, new Shape(2, 1, 1, 3));
            data = DoubleStream.of(data).map(Math::abs).toArray();
            expected = manager.create(data, new Shape(2, 1, 1, 3));
            Assertions.assertAlmostEquals(array.abs(), expected);
            // test scalar
            array = manager.create(-0.00001f);
            expected = manager.create(0.00001f);
            Assertions.assertAlmostEquals(array.abs(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0, 2, 0));
            Assert.assertEquals(array.abs(), array);
        }
    }

    @Test
    public void testSquare() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(x -> Math.pow(x, 2.0)).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.square(), expected);
        }
    }

    @Test
    public void testSqrt() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {2.0, 3.0, 4.0, 5.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::sqrt).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.sqrt(), expected);
            // test multi-dim
            data = new double[] {6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
            array = manager.create(data, new Shape(2, 3));
            data = DoubleStream.of(data).map(Math::sqrt).toArray();
            expected = manager.create(data, new Shape(2, 3));
            Assertions.assertAlmostEquals(array.sqrt(), expected);
            // test scalar
            array = manager.create(4f);
            expected = manager.create(2f);
            Assertions.assertAlmostEquals(array.sqrt(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array.sqrt(), array);
        }
    }

    @Test
    public void testCbrt() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::cbrt).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.cbrt(), expected);
            // test multi-dim
            data = new double[] {1.2, 98.34, 2.34, -0.456, 2, -22};
            array = manager.create(data, new Shape(3, 1, 2, 1));
            data = DoubleStream.of(data).map(Math::cbrt).toArray();
            expected = manager.create(data, new Shape(3, 1, 2, 1));
            Assertions.assertAlmostEquals(array.cbrt(), expected);
            // test scalar
            array = manager.create(125f);
            expected = manager.create(5f);
            Assertions.assertAlmostEquals(array.cbrt(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array.cbrt(), array);
        }
    }

    @Test
    public void testFloor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::floor).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.floor(), expected);
            // test multi-dim
            data = new double[] {2.4, 3.3, 4.5, -2.33, -2.0001, -3.0001};
            array = manager.create(data, new Shape(2, 3));
            data = DoubleStream.of(data).map(Math::floor).toArray();
            expected = manager.create(data, new Shape(2, 3));
            Assertions.assertAlmostEquals(array.floor(), expected);
            // test scalar
            array = manager.create(0.0001f);
            expected = manager.create(0f);
            Assertions.assertAlmostEquals(array.floor(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 1, 2, 0));
            Assert.assertEquals(array.floor(), array);
        }
    }

    @Test
    public void testCeil() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::ceil).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.ceil(), expected);
            // test multi-dim
            data = new double[] {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, 2.3};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::ceil).toArray();
            expected = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(array.ceil(), expected);
            // test scalar
            array = manager.create(1.0001f);
            expected = manager.create(2f);
            Assertions.assertAlmostEquals(array.ceil(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 2, 3, 0));
            Assert.assertEquals(array.ceil(), array);
        }
    }

    @Test
    public void testRound() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::round).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.round(), expected);
            // test multi-dim
            data = new double[] {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, 2.3};
            array = manager.create(data, new Shape(4, 2));
            // the result of round in Maths is different from Numpy
            data = new double[] {-2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0};
            expected = manager.create(data, new Shape(4, 2));
            Assert.assertEquals(array.round(), expected);
            // test scalar
            array = manager.create(1.0001f);
            expected = manager.create(1f);
            Assertions.assertAlmostEquals(array.round(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 2, 3, 0));
            Assert.assertEquals(array.round(), array);
        }
    }

    @Test
    public void testTrunc() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = new double[] {1.0, 2.0, -3, -4, 5, -223};
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.trunc(), expected);
            // test multi-dim
            data = new double[] {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7};
            array = manager.create(data, new Shape(2, 3));
            data = new double[] {-1, -1, 0, 0, 1, 1};
            expected = manager.create(data, new Shape(2, 3));
            Assertions.assertAlmostEquals(array.trunc(), expected);
            // test scalar
            array = manager.create(1.0001f);
            expected = manager.create(1f);
            Assertions.assertAlmostEquals(array.trunc(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 2, 3, 0));
            Assert.assertEquals(array.trunc(), array);
        }
    }

    @Test
    public void testExp() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::exp).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.exp(), expected);
            // test multi-dim
            data = new double[] {2.34, 204.0, 653.222, 1.0};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::exp).toArray();
            expected = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(array.exp(), expected);
            // test scalar
            array = manager.create(2f);
            expected = manager.create(7.389f);
            Assertions.assertAlmostEquals(array.exp(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 3, 3, 2));
            Assert.assertEquals(array.exp(), array);
        }
    }

    @Test
    public void testLog() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::log).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.log(), expected);
            // test multi-dim
            data = new double[] {1, Math.E, Math.E * Math.E};
            array = manager.create(data, new Shape(3, 1, 1, 1));
            data = DoubleStream.of(data).map(Math::log).toArray();
            expected = manager.create(data, new Shape(3, 1, 1, 1));
            Assertions.assertAlmostEquals(array.log(), expected);
            // test scalar
            array = manager.create(Math.E);
            expected = manager.create(1f);
            Assertions.assertAlmostEquals(array.log(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 0));
            Assert.assertEquals(array.log(), array);
        }
    }

    @Test
    public void testLog10() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.12, 3.584, 4.334, 5.111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::log10).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.log10(), expected);
            // test multi-dim
            data = new double[] {1e-15, 1e-14, 1e-13, 1e-2};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::log10).toArray();
            expected = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(array.log10(), expected);
            // test scalar
            array = manager.create(1e-5);
            expected = manager.create(-5.0);
            Assertions.assertAlmostEquals(array.log10(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assertions.assertAlmostEquals(array.log10(), array);
        }
    }

    @Test
    public void testLog2() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(x -> Math.log10(x) / Math.log10(2)).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.log2(), expected);
            // test multi-dim
            data = new double[] {1, 2, 4, 8, 16, 32};
            array = manager.create(data, new Shape(2, 3, 1));
            data = DoubleStream.of(data).map(x -> Math.log10(x) / Math.log10(2)).toArray();
            expected = manager.create(data, new Shape(2, 3, 1));
            Assertions.assertAlmostEquals(array.log2(), expected);
            // test scalar
            array = manager.create(1f);
            expected = manager.create(0f);
            Assertions.assertAlmostEquals(array.log2(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assertions.assertAlmostEquals(array.log2(), array);
        }
    }

    @Test
    public void testSin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::sin).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.sin(), expected);
            // test multi-dim
            data = new double[] {0, 30, 45, 60, 90, 120};
            array = manager.create(data, new Shape(2, 3));
            data = DoubleStream.of(data).map(Math::sin).toArray();
            expected = manager.create(data, new Shape(2, 3));
            Assert.assertEquals(array.sin(), expected);
            // test scalar
            array = manager.create(0.5 * Math.PI);
            expected = manager.create(1.0);
            Assert.assertEquals(array.sin(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array.sin(), array);
        }
    }

    @Test
    public void testCos() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::cos).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.cos(), expected);
            // test multi-dim
            data = new double[] {0, Math.PI / 4, Math.PI / 2, Math.PI};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::cos).toArray();
            expected = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(array.cos(), expected);
            // test scalar
            array = manager.create(0f);
            expected = manager.create(1.0f);
            Assert.assertEquals(array.cos(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 1, 2));
            Assert.assertEquals(array.cos(), array);
        }
    }

    @Test
    public void testTan() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0.0, Math.PI / 4.0, Math.PI / 2.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::tan).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.tan(), expected);
            // test multi-dim
            data = new double[] {0, -Math.PI, Math.PI / 2.0, Math.PI};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::tan).toArray();
            expected = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(array.tan(), expected);
            // test scalar
            array = manager.create(0f);
            expected = manager.create(0f);
            Assert.assertEquals(array.tan(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0, 2));
            Assert.assertEquals(array.tan(), array);
        }
    }

    @Test
    public void testAsin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, -1.0, -0.22, 0.4, 0.1234};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::asin).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.asin(), expected);
            // test multi-dim
            data = new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
            array = manager.create(data, new Shape(2, 3, 1));
            data = DoubleStream.of(data).map(Math::asin).toArray();
            expected = manager.create(data, new Shape(2, 3, 1));
            Assertions.assertAlmostEquals(array.asin(), expected);
            // test scalar
            array = manager.create(0f);
            expected = manager.create(0f);
            Assert.assertEquals(array.asin(), expected);
            // test zero-dim
            array = manager.create(new Shape(2, 0, 2));
            Assert.assertEquals(array.asin(), array);
        }
    }

    @Test
    public void testAcos() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {-1.0, -0.707, 0.0, 0.707, 1.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::acos).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.acos(), expected);
            // test multi-dim
            data = new double[] {-1.0, -0.707, -0.5, 0, 0.5, 0.707, 1.0};
            array = manager.create(data, new Shape(7, 1));
            data = DoubleStream.of(data).map(Math::acos).toArray();
            expected = manager.create(data, new Shape(7, 1));
            Assertions.assertAlmostEquals(array.acos(), expected);
            // test scalar
            array = manager.create(0f);
            expected = manager.create(Math.PI / 2);
            Assertions.assertAlmostEquals(array.acos(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array.acos(), array);
        }
    }

    @Test
    public void testAtan() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {-1.0, 0.0, 1.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::atan).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.atan(), expected);
            // test multi-dim
            data = new double[] {-1.0, -0.5, 0, 0.5, 1.0};
            array = manager.create(data, new Shape(5, 1));
            data = DoubleStream.of(data).map(Math::atan).toArray();
            expected = manager.create(data, new Shape(5, 1));
            Assertions.assertAlmostEquals(array.atan(), expected);
            // test scalar
            array = manager.create(0f);
            expected = manager.create(0f);
            Assertions.assertAlmostEquals(array.atan(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 0));
            Assert.assertEquals(array.atan(), array);
        }
    }

    @Test
    public void testToDegrees() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::toDegrees).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.toDegrees(), expected);
            // test multi-dim
            data =
                    new double[] {
                        0, Math.PI / 6, Math.PI / 3, 2 * Math.PI / 3, 5.0 / 6 * Math.PI, Math.PI
                    };
            array = manager.create(data, new Shape(2, 1, 3));
            data = DoubleStream.of(data).map(Math::toDegrees).toArray();
            expected = manager.create(data, new Shape(2, 1, 3));
            Assertions.assertAlmostEquals(array.toDegrees(), expected);
            // test scalar
            array = manager.create(Math.PI);
            expected = manager.create(180f);
            Assertions.assertAlmostEquals(array.toDegrees(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array.toDegrees(), array);
        }
    }

    @Test
    public void testToRadians() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0.0, 90.0, 180.0, 270.0, 360.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::toRadians).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.toRadians(), expected);
            // test multi-dim
            data =
                    new double[] {
                        30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
                        360.0
                    };
            array = manager.create(data, new Shape(2, 2, 3));
            data = DoubleStream.of(data).map(Math::toRadians).toArray();
            expected = manager.create(data, new Shape(2, 2, 3));
            Assertions.assertAlmostEquals(array.toRadians(), expected);
            // test scalar
            array = manager.create(180f);
            expected = manager.create(Math.PI);
            Assertions.assertAlmostEquals(array.toRadians(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 1, 0, 1));
            Assert.assertEquals(array.toRadians(), array);
        }
    }

    @Test
    public void testSinh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::sinh).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.sinh(), expected);
            // test multi-dim
            data = new double[] {0.0, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::sinh).toArray();
            expected = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(array.sinh(), expected);
            // test scalar
            array = manager.create(5f);
            expected = manager.create(74.2032f);
            Assertions.assertAlmostEquals(array.sinh(), expected);
            // test zero-dim
            array = manager.create(new Shape(1, 0, 0, 1));
            Assert.assertEquals(array.sinh(), array);
        }
    }

    @Test
    public void testCosh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::cosh).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.cosh(), expected);
            // test multi-dim
            data = new double[] {0.0, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::cosh).toArray();
            expected = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(array.cosh(), expected);
            // test scalar
            array = manager.create(5f);
            expected = manager.create(74.21f);
            Assertions.assertAlmostEquals(array.cosh(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0, 0, 0));
            Assert.assertEquals(array.cosh(), array);
        }
    }

    @Test
    public void testTanh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::tanh).toArray();
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.tanh(), expected);
            // test multi-dim
            data = new double[] {0.0, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::tanh).toArray();
            expected = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(array.tanh(), expected);
            // test scalar
            array = manager.create(5f);
            expected = manager.create(0.9999f);
            Assertions.assertAlmostEquals(array.tanh(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 4, 0, 0));
            Assert.assertEquals(array.tanh(), array);
        }
    }

    @Test
    public void testAsinh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {Math.E, 10.0};
            NDArray array = manager.create(data);
            data = new double[] {1.72538256, 2.99822295};
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.asinh(), expected);
            // test multi-dim
            array = manager.arange(10.0f).reshape(5, 1, 2);
            expected =
                    manager.create(
                            new float[] {
                                0f, 0.88137f, 1.44364f, 1.81845f, 2.0947f, 2.3124f, 2.49178f,
                                2.64412f, 2.77648f, 2.89344f
                            },
                            new Shape(5, 1, 2));
            Assertions.assertAlmostEquals(array.asinh(), expected);
            // test scalar
            array = manager.create(0f);
            expected = manager.create(0f);
            Assertions.assertAlmostEquals(array.asinh(), expected);
            // test zero-dim
            array = manager.create(new Shape(0));
            Assert.assertEquals(array.asinh(), array);
        }
    }

    @Test
    public void testAcosh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {Math.E, 10.0};
            NDArray array = manager.create(data);
            data = new double[] {1.65745445, 2.99322285};
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.acosh(), expected);
            // test multi-dim
            array = manager.arange(10.0f, 110.0f, 10.0f).reshape(2, 5);
            expected =
                    manager.create(
                            new float[] {
                                2.9932f, 3.68825f, 4.0941f, 4.38188f, 4.6051f, 4.7874f, 4.9416f,
                                5.07513f, 5.193f, 5.2983f
                            },
                            new Shape(2, 5));
            Assertions.assertAlmostEquals(array.acosh(), expected);
            // test scalar
            array = manager.create(1f);
            expected = manager.create(0f);
            Assertions.assertAlmostEquals(array.acosh(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assert.assertEquals(array.acosh(), array);
        }
    }

    @Test
    public void testAtanh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0.0, -0.5};
            NDArray array = manager.create(data);
            data = new double[] {0.0, -0.54930614};
            NDArray expected = manager.create(data);
            Assertions.assertAlmostEquals(array.atanh(), expected);
            // test multi-dim
            array = manager.create(new float[] {0.0f, 0.1f, 0.2f, 0.3f}, new Shape(2, 2));
            expected =
                    manager.create(new float[] {0.0f, 0.10033f, 0.2027f, 0.3095f}, new Shape(2, 2));
            Assertions.assertAlmostEquals(array.atanh(), expected);
            // test scalar
            array = manager.create(0.5f);
            expected = manager.create(0.5493f);
            Assertions.assertAlmostEquals(array.atanh(), expected);
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assert.assertEquals(array.atanh(), array);
        }
    }

    @Test
    public void testSign() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {-3, 5, -7};
            NDArray array = manager.create(data);
            data = new double[] {-1, 1, -1};
            NDArray expected = manager.create(data);
            Assert.assertEquals(array.sign(), expected);
            Assert.assertNotEquals(array, expected);

            array = manager.create(new double[] {0, -7, -8, 2, -9});
            expected = manager.create(new double[] {0, -1, -1, 1, -1});
            Assert.assertEquals(array.signi(), expected);
            Assert.assertEquals(array, expected);

            // test multi-dim
            array = manager.create(new float[] {1f, -1f, 0f, 7f}, new Shape(2, 2));
            expected = manager.create(new float[] {1f, -1f, 0f, 1f}, new Shape(2, 2));
            Assertions.assertAlmostEquals(array.sign(), expected);

            // test scalar
            array = manager.create(0.5f);
            expected = manager.create(1f);
            Assertions.assertAlmostEquals(array.sign(), expected);

            array = manager.create(0f);
            Assertions.assertAlmostEquals(array.sign(), array);

            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assert.assertEquals(array.sign(), array);
        }
    }
}
