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

import java.util.stream.DoubleStream;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayNumericOpTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", NDArrayNumericOpTest.class.getName()};
        new IntegrationTest().runTests(cmd);
    }

    @RunAsTest
    public void testNegation() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray ndArray = manager.create(new float[] {6, 9, -12, -11, 0}, new Shape(1, 5));
            NDArray result = ndArray.neg();
            NDArray inPlaceResult = ndArray.negi();
            NDArray solution = manager.create(new float[] {-6, -9, 12, 11, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Scalar subtraction: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    ndArray, inPlaceResult, "Scalar subtraction: In-place operation failed");
        }
    }

    @RunAsTest
    public void testAbs() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::abs).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.abs(), expectedND);
        }
    }

    @RunAsTest
    public void testSquare() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(x -> Math.pow(x, 2.0)).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.square(), expectedND);
        }
    }

    @RunAsTest
    public void testCbrt() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData =
                    new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::cbrt).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.cbrt(), expectedND);
        }
    }

    @RunAsTest
    public void testFloor() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData =
                    new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::floor).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.floor(), expectedND);
        }
    }

    @RunAsTest
    public void testCeil() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData =
                    new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::ceil).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.ceil(), expectedND);
        }
    }

    @RunAsTest
    public void testRound() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData =
                    new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::round).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.round(), expectedND);
        }
    }

    @RunAsTest
    public void testTrunc() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData =
                    new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray testedND = manager.create(testedData);
            double[] truncData = new double[] {1.0, 2.0, -3, -4, 5, -223};
            NDArray expectedND = manager.create(truncData);
            Assertions.assertAlmostEquals(testedND.trunc(), expectedND);
        }
    }

    @RunAsTest
    public void testExp() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData =
                    new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::exp).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.exp(), expectedND);
        }
    }

    @RunAsTest
    public void testLog() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::log).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.log(), expectedND);
        }
    }

    @RunAsTest
    public void testLog10() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::log10).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.log10(), expectedND);
        }
    }

    @RunAsTest
    public void testLog2() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData =
                    DoubleStream.of(testedData).map(x -> Math.log10(x) / Math.log10(2)).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.log2(), expectedND);
        }
    }

    @RunAsTest
    public void testSin() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::sin).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.sin(), expectedND);
        }
    }

    @RunAsTest
    public void testCos() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::cos).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.cos(), expectedND);
        }
    }

    @RunAsTest
    public void testTan() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {0.0, Math.PI / 4.0, Math.PI / 2.0};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::tan).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.tan(), expectedND);
        }
    }

    @RunAsTest
    public void testAsin() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, -1.0, -0.22, 0.4, 0.1234};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::asin).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.asin(), expectedND);
        }
    }

    @RunAsTest
    public void testAcos() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {-1.0, -0.707, 0.0, 0.707, 1.0};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::acos).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.acos(), expectedND);
        }
    }

    @RunAsTest
    public void testAtan() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {-1.0, 0.0, 1.0};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::atan).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.atan(), expectedND);
        }
    }

    @RunAsTest
    public void testToDegrees() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData =
                    new double[] {0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::toDegrees).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.toDegrees(), expectedND);
        }
    }

    @RunAsTest
    public void testToRadians() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {0.0, 90.0, 180.0, 270.0, 360.0};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::toRadians).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.toRadians(), expectedND);
        }
    }

    @RunAsTest
    public void testSinh() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::sinh).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.sinh(), expectedND);
        }
    }

    @RunAsTest
    public void testCosh() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::cosh).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.cosh(), expectedND);
        }
    }

    @RunAsTest
    public void testTanh() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
            NDArray testedND = manager.create(testedData);
            testedData = DoubleStream.of(testedData).map(Math::tanh).toArray();
            NDArray expectedND = manager.create(testedData);
            Assertions.assertAlmostEquals(testedND.tanh(), expectedND);
        }
    }

    @RunAsTest
    public void testAsinh() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {Math.E, 10.0};
            NDArray testedND = manager.create(testedData);
            double[] aSinhData = new double[] {1.72538256, 2.99822295};
            NDArray expectedND = manager.create(aSinhData);
            Assertions.assertAlmostEquals(testedND.asinh(), expectedND);
        }
    }

    @RunAsTest
    public void testAtanh() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            double[] testedData = new double[] {0.0, -0.5};
            NDArray testedND = manager.create(testedData);
            double[] aTanhData = new double[] {0.0, -0.54930614};
            NDArray expectedND = manager.create(aTanhData);
            Assertions.assertAlmostEquals(testedND.atanh(), expectedND);
        }
    }
}
