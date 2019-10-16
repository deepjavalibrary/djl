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

import ai.djl.integration.util.Assertions;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
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
            NDArray actual = manager.create(data);
            Assert.assertEquals(actual, array.neg());
            Assertions.assertInPlaceEquals(actual, array.negi(), array);
            // test multi-dim
            data = new double[] {-2.2, 2.2, 3, -0.2, 2.76, 0.0002};
            array = manager.create(data, new Shape(2, 3));
            data = DoubleStream.of(data).map(x -> -x).toArray();
            actual = manager.create(data, new Shape(2, 3));
            Assert.assertEquals(actual, array.neg());
            Assertions.assertInPlaceEquals(actual, array.negi(), array);
            // test scalar
            array = manager.create(3f);
            actual = manager.create(-3f);
            Assert.assertEquals(actual, array.neg());
            Assertions.assertInPlaceEquals(actual, array.negi(), array);
            // test zero-dim
            array = manager.create(new Shape(2, 0, 1));
            actual = manager.create(new Shape(2, 0, 1));
            Assert.assertEquals(actual, array.neg());
            Assertions.assertInPlaceEquals(actual, array.negi(), array);
        }
    }

    @Test
    public void testAbs() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::abs).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.abs());
            // test multi-dim
            data = new double[] {1.2, 98.34, 2.34, -0.456, 2, -22};
            array = manager.create(data, new Shape(2, 1, 1, 3));
            data = DoubleStream.of(data).map(Math::abs).toArray();
            actual = manager.create(data, new Shape(2, 1, 1, 3));
            Assertions.assertAlmostEquals(actual, array.abs());
            // test scalar
            array = manager.create(0.00001f);
            actual = manager.create(-0.00001f);
            Assertions.assertAlmostEquals(actual, array.abs());
            // test zero-dim
            array = manager.create(new Shape(0, 0, 2, 0));
            Assert.assertEquals(array, array.abs());
        }
    }

    @Test
    public void testSquare() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(x -> Math.pow(x, 2.0)).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.square());
            // test multi-dim
            data = new double[] {1.2, 98.34, 2.34, -0.456, 2, -22};
            array = manager.create(data, new Shape(2, 1, 1, 3));
            data = DoubleStream.of(data).map(Math::abs).toArray();
            actual = manager.create(data, new Shape(2, 1, 1, 3));
            Assertions.assertAlmostEquals(actual, array.abs());
            // test scalar
            array = manager.create(0.00001f);
            actual = manager.create(-0.00001f);
            Assertions.assertAlmostEquals(actual, array.abs());
            // test zero-dim
            array = manager.create(new Shape(0, 0, 2, 0));
            Assert.assertEquals(array, array.abs());
        }
    }

    @Test
    public void testCbrt() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::cbrt).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.cbrt());
            // test multi-dim
            data = new double[] {1.2, 98.34, 2.34, -0.456, 2, -22};
            array = manager.create(data, new Shape(3, 1, 2, 1));
            data = DoubleStream.of(data).map(Math::cbrt).toArray();
            actual = manager.create(data, new Shape(3, 1, 2, 1));
            Assertions.assertAlmostEquals(actual, array.cbrt());
            // test scalar
            array = manager.create(125f);
            actual = manager.create(5f);
            Assertions.assertAlmostEquals(actual, array.cbrt());
            // test zero-dim
            array = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array, array.cbrt());
        }
    }

    @Test
    public void testFloor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::floor).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.floor());
            // test multi-dim
            data = new double[] {2.4, 3.3, 4.5, -2.33, -2.0001, -3.0001};
            array = manager.create(data, new Shape(2, 3));
            data = DoubleStream.of(data).map(Math::floor).toArray();
            actual = manager.create(data, new Shape(2, 3));
            Assertions.assertAlmostEquals(actual, array.floor());
            // test scalar
            array = manager.create(0.0001f);
            actual = manager.create(0f);
            Assertions.assertAlmostEquals(actual, array.floor());
            // test zero-dim
            array = manager.create(new Shape(1, 1, 2, 0));
            Assert.assertEquals(array, array.floor());
        }
    }

    @Test
    public void testCeil() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::ceil).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.ceil());
            // test multi-dim
            data = new double[] {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, 2.3};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::ceil).toArray();
            actual = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(actual, array.ceil());
            // test scalar
            array = manager.create(1.0001f);
            actual = manager.create(2f);
            Assertions.assertAlmostEquals(actual, array.ceil());
            // test zero-dim
            array = manager.create(new Shape(1, 2, 3, 0));
            Assert.assertEquals(array, array.ceil());
        }
    }

    @Test
    public void testRound() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::round).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.round());
            // test multi-dim
            data = new double[] {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, 2.3};
            array = manager.create(data, new Shape(4, 2));
            // the result of round in Maths is different from Numpy
            data = new double[] {-2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0};
            actual = manager.create(data, new Shape(4, 2));
            Assert.assertEquals(actual, array.round());
            // test scalar
            array = manager.create(1.0001f);
            actual = manager.create(1f);
            Assertions.assertAlmostEquals(actual, array.round());
            // test zero-dim
            array = manager.create(new Shape(1, 2, 3, 0));
            Assert.assertEquals(array, array.round());
        }
    }

    @Test
    public void testTrunc() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = new double[] {1.0, 2.0, -3, -4, 5, -223};
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.trunc());
            // test multi-dim
            data = new double[] {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7};
            array = manager.create(data, new Shape(2, 3));
            data = new double[] {-1, -1, 0, 0, 1, 1};
            actual = manager.create(data, new Shape(2, 3));
            Assertions.assertAlmostEquals(actual, array.trunc());
            // test scalar
            array = manager.create(1.0001f);
            actual = manager.create(1f);
            Assertions.assertAlmostEquals(actual, array.trunc());
            // test zero-dim
            array = manager.create(new Shape(1, 2, 3, 0));
            Assert.assertEquals(array, array.trunc());
        }
    }

    @Test
    public void testExp() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::exp).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.exp());
            // test multi-dim
            data = new double[] {2.34, 204.0, 653.222, 1.0};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::exp).toArray();
            actual = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(actual, array.exp());
            // test scalar
            array = manager.create(2f);
            actual = manager.create(7.389f);
            Assertions.assertAlmostEquals(actual, array.exp());
            // test zero-dim
            array = manager.create(new Shape(0, 3, 3, 2));
            Assert.assertEquals(array, array.exp());
        }
    }

    @Test
    public void testLog() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::log).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.log());
            // test multi-dim
            data = new double[] {1, Math.E, Math.E * Math.E};
            array = manager.create(data, new Shape(3, 1, 1, 1));
            data = DoubleStream.of(data).map(Math::log).toArray();
            actual = manager.create(data, new Shape(3, 1, 1, 1));
            Assertions.assertAlmostEquals(actual, array.log());
            // test scalar
            array = manager.create(Math.E);
            actual = manager.create(1f);
            Assertions.assertAlmostEquals(actual, array.log());
            // test zero-dim
            array = manager.create(new Shape(1, 0));
            Assert.assertEquals(array, array.log());
        }
    }

    @Test
    public void testLog10() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.12, 3.584, 4.334, 5.111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::log10).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.log10());
            // test multi-dim
            data = new double[] {1e-15, 1e-14, 1e-13, 1e-2};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::log10).toArray();
            actual = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(actual, array.log10());
            // test scalar
            array = manager.create(1e-5);
            actual = manager.create(-5.0);
            Assertions.assertAlmostEquals(actual, array.log10());
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assertions.assertAlmostEquals(array, array.log10());
        }
    }

    @Test
    public void testLog2() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(x -> Math.log10(x) / Math.log10(2)).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.log2());
            // test multi-dim
            data = new double[] {1, 2, 4, 8, 16, 32};
            array = manager.create(data, new Shape(2, 3, 1));
            data = DoubleStream.of(data).map(x -> Math.log10(x) / Math.log10(2)).toArray();
            actual = manager.create(data, new Shape(2, 3, 1));
            Assertions.assertAlmostEquals(actual, array.log2());
            // test scalar
            array = manager.create(1f);
            actual = manager.create(0f);
            Assertions.assertAlmostEquals(actual, array.log2());
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assertions.assertAlmostEquals(array, array.log2());
        }
    }

    @Test
    public void testSin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::sin).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.sin());
            // test multi-dim
            data = new double[] {0, 30, 45, 60, 90, 120};
            array = manager.create(data, new Shape(2, 3));
            data = DoubleStream.of(data).map(Math::sin).toArray();
            actual = manager.create(data, new Shape(2, 3));
            Assert.assertEquals(actual, array.sin());
            // test scalar
            array = manager.create(0.5 * Math.PI);
            actual = manager.create(1.0);
            Assert.assertEquals(actual, array.sin());
            // test zero-dim
            array = manager.create(new Shape(1, 0, 2));
            Assert.assertEquals(array, array.sin());
        }
    }

    @Test
    public void testCos() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::cos).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.cos());
            // test multi-dim
            data = new double[] {0, Math.PI / 4, Math.PI / 2, Math.PI};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::cos).toArray();
            actual = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(actual, array.cos());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(1.0f);
            Assert.assertEquals(actual, array.cos());
            // test zero-dim
            array = manager.create(new Shape(0, 1, 2));
            Assert.assertEquals(array, array.cos());
        }
    }

    @Test
    public void testTan() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0.0, Math.PI / 4.0, Math.PI / 2.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::tan).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.tan());
            // test multi-dim
            data = new double[] {0, -Math.PI, Math.PI / 2.0, Math.PI};
            array = manager.create(data, new Shape(2, 2));
            data = DoubleStream.of(data).map(Math::tan).toArray();
            actual = manager.create(data, new Shape(2, 2));
            Assertions.assertAlmostEquals(actual, array.tan());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(0f);
            Assert.assertEquals(actual, array.tan());
            // test zero-dim
            array = manager.create(new Shape(0, 0, 2));
            Assert.assertEquals(array, array.tan());
        }
    }

    @Test
    public void testAsin() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, -1.0, -0.22, 0.4, 0.1234};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::asin).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.asin());
            // test multi-dim
            data = new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
            array = manager.create(data, new Shape(2, 3, 1));
            data = DoubleStream.of(data).map(Math::asin).toArray();
            actual = manager.create(data, new Shape(2, 3, 1));
            Assertions.assertAlmostEquals(actual, array.asin());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(0f);
            Assert.assertEquals(actual, array.asin());
            // test zero-dim
            array = manager.create(new Shape(2, 0, 2));
            Assert.assertEquals(array, array.asin());
        }
    }

    @Test
    public void testAcos() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {-1.0, -0.707, 0.0, 0.707, 1.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::acos).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.acos());
            // test multi-dim
            data = new double[] {-1.0, -0.707, -0.5, 0, 0.5, 0.707, 1.0};
            array = manager.create(data, new Shape(7, 1));
            data = DoubleStream.of(data).map(Math::acos).toArray();
            actual = manager.create(data, new Shape(7, 1));
            Assertions.assertAlmostEquals(actual, array.acos());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(Math.PI / 2);
            Assertions.assertAlmostEquals(actual, array.acos());
            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array, array.acos());
        }
    }

    @Test
    public void testAtan() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {-1.0, 0.0, 1.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::atan).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.atan());
            // test multi-dim
            data = new double[] {-1.0, -0.5, 0, 0.5, 1.0};
            array = manager.create(data, new Shape(5, 1));
            data = DoubleStream.of(data).map(Math::atan).toArray();
            actual = manager.create(data, new Shape(5, 1));
            Assertions.assertAlmostEquals(actual, array.atan());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(0f);
            Assertions.assertAlmostEquals(actual, array.atan());
            // test zero-dim
            array = manager.create(new Shape(1, 0));
            Assert.assertEquals(array, array.atan());
        }
    }

    @Test
    public void testToDegrees() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::toDegrees).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.toDegrees());
            // test multi-dim
            data =
                    new double[] {
                        0, Math.PI / 6, Math.PI / 3, 2 * Math.PI / 3, 5.0 / 6 * Math.PI, Math.PI
                    };
            array = manager.create(data, new Shape(2, 1, 3));
            data = DoubleStream.of(data).map(Math::toDegrees).toArray();
            actual = manager.create(data, new Shape(2, 1, 3));
            Assertions.assertAlmostEquals(actual, array.toDegrees());
            // test scalar
            array = manager.create(Math.PI);
            actual = manager.create(180f);
            Assertions.assertAlmostEquals(actual, array.toDegrees());
            // test zero-dim
            array = manager.create(new Shape(0, 1));
            Assert.assertEquals(array, array.toDegrees());
        }
    }

    @Test
    public void testToRadians() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0.0, 90.0, 180.0, 270.0, 360.0};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::toRadians).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.toRadians());
            // test multi-dim
            data =
                    new double[] {
                        30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
                        360.0
                    };
            array = manager.create(data, new Shape(2, 2, 3));
            data = DoubleStream.of(data).map(Math::toRadians).toArray();
            actual = manager.create(data, new Shape(2, 2, 3));
            Assertions.assertAlmostEquals(actual, array.toRadians());
            // test scalar
            array = manager.create(180f);
            actual = manager.create(Math.PI);
            Assertions.assertAlmostEquals(actual, array.toRadians());
            // test zero-dim
            array = manager.create(new Shape(1, 1, 0, 1));
            Assert.assertEquals(array, array.toRadians());
        }
    }

    @Test
    public void testSinh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::sinh).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.sinh());
            // test multi-dim
            data = new double[] {0.0, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::sinh).toArray();
            actual = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(actual, array.sinh());
            // test scalar
            array = manager.create(5f);
            actual = manager.create(74.2032f);
            Assertions.assertAlmostEquals(actual, array.sinh());
            // test zero-dim
            array = manager.create(new Shape(1, 0, 0, 1));
            Assert.assertEquals(array, array.sinh());
        }
    }

    @Test
    public void testCosh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::cosh).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.cosh());
            // test multi-dim
            data = new double[] {0.0, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::cosh).toArray();
            actual = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(actual, array.cosh());
            // test scalar
            array = manager.create(5f);
            actual = manager.create(74.21f);
            Assertions.assertAlmostEquals(actual, array.cosh());
            // test zero-dim
            array = manager.create(new Shape(0, 0, 0, 0));
            Assert.assertEquals(array, array.cosh());
        }
    }

    @Test
    public void testTanh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray array = manager.create(data);
            data = DoubleStream.of(data).map(Math::tanh).toArray();
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.tanh());
            // test multi-dim
            data = new double[] {0.0, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77};
            array = manager.create(data, new Shape(2, 2, 2));
            data = DoubleStream.of(data).map(Math::tanh).toArray();
            actual = manager.create(data, new Shape(2, 2, 2));
            Assertions.assertAlmostEquals(actual, array.tanh());
            // test scalar
            array = manager.create(5f);
            actual = manager.create(0.9999f);
            Assertions.assertAlmostEquals(actual, array.tanh());
            // test zero-dim
            array = manager.create(new Shape(0, 4, 0, 0));
            Assert.assertEquals(array, array.tanh());
        }
    }

    @Test
    public void testAsinh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {Math.E, 10.0};
            NDArray array = manager.create(data);
            data = new double[] {1.72538256, 2.99822295};
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.asinh());
            // test multi-dim
            array = manager.arange(10).reshape(5, 1, 2);
            actual =
                    manager.create(
                            new float[] {
                                0f, 0.88137f, 1.44364f, 1.81845f, 2.0947f, 2.3124f, 2.49178f,
                                2.64412f, 2.77648f, 2.89344f
                            },
                            new Shape(5, 1, 2));
            Assertions.assertAlmostEquals(actual, array.asinh());
            // test scalar
            array = manager.create(0f);
            actual = manager.create(0f);
            Assertions.assertAlmostEquals(actual, array.asinh());
            // test zero-dim
            array = manager.create(new Shape(0));
            Assert.assertEquals(array, array.asinh());
        }
    }

    @Test
    public void testAcosh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {Math.E, 10.0};
            NDArray array = manager.create(data);
            data = new double[] {1.65745445, 2.99322285};
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.acosh());
            // test multi-dim
            array = manager.arange(10, 110, 10).reshape(2, 5);
            actual =
                    manager.create(
                            new float[] {
                                2.9932f, 3.68825f, 4.0941f, 4.38188f, 4.6051f, 4.7874f, 4.9416f,
                                5.07513f, 5.193f, 5.2983f
                            },
                            new Shape(2, 5));
            Assertions.assertAlmostEquals(actual, array.acosh());
            // test scalar
            array = manager.create(1f);
            actual = manager.create(0f);
            Assertions.assertAlmostEquals(actual, array.acosh());
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assert.assertEquals(array, array.acosh());
        }
    }

    @Test
    public void testAtanh() {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] data = {0.0, -0.5};
            NDArray array = manager.create(data);
            data = new double[] {0.0, -0.54930614};
            NDArray actual = manager.create(data);
            Assertions.assertAlmostEquals(actual, array.atanh());
            // test multi-dim
            array = manager.create(new float[] {0.0f, 0.1f, 0.2f, 0.3f}, new Shape(2, 2));
            actual =
                    manager.create(new float[] {0.0f, 0.10033f, 0.2027f, 0.3095f}, new Shape(2, 2));
            Assertions.assertAlmostEquals(actual, array.atanh());
            // test scalar
            array = manager.create(0.5f);
            actual = manager.create(0.5493f);
            Assertions.assertAlmostEquals(actual, array.atanh());
            // test zero-dim
            array = manager.create(new Shape(0, 0));
            Assert.assertEquals(array, array.atanh());
        }
    }
}
