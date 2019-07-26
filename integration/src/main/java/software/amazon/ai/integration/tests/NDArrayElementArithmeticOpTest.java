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

import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class NDArrayElementArithmeticOpTest extends AbstractTest {

    public static void main(String[] args) {
        new NDArrayElementArithmeticOpTest().runTest(args);
    }

    @RunAsTest
    public void testAddScalar() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray lhs = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 4));
            NDArray result;
            try (MxAutograd autograd = new MxAutograd()) {
                autograd.attachGradient(lhs);
                MxAutograd.setRecording(true);
                result = NDArrays.add(lhs, 2);
                autograd.backward((MxNDArray) result);
            }
            // check add scalar result

            Assertions.assertFalse(
                    NDArrays.equals(lhs, result),
                    "None in-place operator returned in-place result");
            NDArray expected = manager.create(new float[] {3f, 4f, 5f, 6f}, new Shape(1, 4));
            Assertions.assertEquals(expected, result, "AddScala: Incorrect value in summed array");

            // check add backward
            NDArray expectedGradient =
                    manager.create(new float[] {1f, 1f, 1f, 1f}, new Shape(1, 4));
            Assertions.assertEquals(
                    expectedGradient,
                    lhs.getGradient(),
                    "AddScala backward: Incorrect gradient after backward");
        }
    }

    @RunAsTest
    public void testAddScalarInPlace() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray addend = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 4));
            NDArray result = NDArrays.addi(addend, 2);
            Assertions.assertInPlace(result, addend, "In-place summation failed");
            NDArray solution = manager.create(new float[] {3f, 4f, 5f, 6f}, new Shape(1, 4));
            Assertions.assertEquals(solution, result, "Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testAddNDArray() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray addend = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 4));
            NDArray addendum = manager.create(new float[] {2f, 3f, 4f, 5f}, new Shape(1, 4));
            NDArray result = NDArrays.add(addend, addendum);
            Assertions.assertFalse(
                    NDArrays.equals(addend, result),
                    "None in-place operator returned in-place result");
            NDArray solution = manager.create(new float[] {3f, 5f, 7f, 9f}, new Shape(1, 4));
            Assertions.assertEquals(solution, result, "Incorrect value in summed array");

            NDArray[] toAddAll =
                    new NDArray[] {
                        manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2)),
                        manager.create(new float[] {4, 3, 2, 1}, new Shape(2, 2)),
                        manager.create(new float[] {2, 2, 2, 2}, new Shape(2, 2))
                    };
            NDArray addAll = NDArrays.add(toAddAll);
            Assertions.assertFalse(
                    addAll.equals(toAddAll[0]), "None in-place operator returned in-place result");
            NDArray addAllResult = manager.create(new float[] {7, 7, 7, 7}, new Shape(2, 2));
            Assertions.assertEquals(addAllResult, addAll, "Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testAddNDArrayInPlace() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray addend = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 4));
            NDArray addendum = manager.create(new float[] {2f, 3f, 4f, 5f}, new Shape(1, 4));
            NDArray result = NDArrays.addi(addend, addendum);
            Assertions.assertInPlace(result, addend, "In-place summation failed");
            NDArray solution = manager.create(new float[] {3f, 5f, 7f, 9f}, new Shape(1, 4));
            Assertions.assertEquals(solution, result, "Incorrect value in summed array");

            NDArray[] toAddAll =
                    new NDArray[] {
                        manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2)),
                        manager.create(new float[] {4, 3, 2, 1}, new Shape(2, 2)),
                        manager.create(new float[] {2, 2, 2, 2}, new Shape(2, 2))
                    };
            NDArray addAll = NDArrays.addi(toAddAll);
            Assertions.assertTrue(addAll.equals(toAddAll[0]), "In-place summation failed");
            NDArray addAllResult = manager.create(new float[] {7, 7, 7, 7}, new Shape(2, 2));
            Assertions.assertEquals(addAllResult, addAll, "Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testScalarSubtraction() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 9, 12, 11, 0}, new Shape(1, 5));
            NDArray result = NDArrays.sub(minuend, 3);
            NDArray inPlaceResult = NDArrays.subi(minuend, 3);
            NDArray solution = manager.create(new float[] {3, 6, 9, 8, -3}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Scalar subtraction: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    minuend, inPlaceResult, "Scalar subtraction: In-place operation failed");
        }
    }

    @RunAsTest
    public void testElemWiseSubtraction() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 9, 12, 15, 0}, new Shape(1, 5));
            NDArray subtrahend = manager.create(new float[] {2, 3, 4, 5, 6}, new Shape(1, 5));
            NDArray result = NDArrays.sub(minuend, subtrahend);
            NDArray inPlaceResult = NDArrays.subi(minuend, subtrahend);
            NDArray solution = manager.create(new float[] {4, 6, 8, 10, -6}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution,
                    result,
                    "Element wise subtraction: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    minuend, inPlaceResult, "Element wise subtraction: In-place operation failed");
        }
    }

    @RunAsTest
    public void testReverseScalarSubtraction() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 91, 12, 215, 180}, new Shape(1, 5));
            NDArray result = NDArrays.sub(180, minuend);
            NDArray inPlaceResult = NDArrays.subi(180, minuend);
            NDArray solution = manager.create(new float[] {174, 89, 168, -35, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution,
                    result,
                    "Scalar reverse subtraction: Incorrect value in result ndarray");
            Assertions.assertTrue(
                    NDArrays.equals(solution, inPlaceResult),
                    "Scalar in-place reverse subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    minuend,
                    inPlaceResult,
                    "Scalar reverse subtraction: In-place operation failed");
        }
    }

    @RunAsTest
    public void testReverseElemWiseSubtraction() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 9, 12, 15, 45}, new Shape(1, 5));
            NDArray subtrahend = manager.create(new float[] {24, 63, 96, 15, 90}, new Shape(1, 5));
            NDArray result = minuend.getNDArrayInternal().rsub(subtrahend);
            NDArray inPlaceResult = minuend.getNDArrayInternal().rsubi(subtrahend);
            NDArray solution = manager.create(new float[] {18, 54, 84, 0, 45}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution,
                    result,
                    "Reverse Element wise subtraction: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Reverse Element wise in-place subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    minuend,
                    inPlaceResult,
                    "Reverse Element wise subtraction: In-place operation failed");
        }
    }

    @RunAsTest
    public void testScalarMultiplication() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray multiplicand = manager.create(new float[] {6, 9, -12, 15, 0}, new Shape(1, 5));
            NDArray result = NDArrays.mul(multiplicand, 3);
            NDArray inPlaceResult = NDArrays.muli(multiplicand, 3);
            NDArray solution = manager.create(new float[] {18, 27, -36, 45, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Scalar multiplication: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place multiplication: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    multiplicand,
                    inPlaceResult,
                    "Scalar multiplication: In-place operation failed");
        }
    }

    @RunAsTest
    public void testElemWiseMultiplication() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray multiplicand = manager.create(new float[] {6, 9, 12, 15, 0}, new Shape(1, 5));
            NDArray with = manager.create(new float[] {2, 3, 4, 5, 6}, new Shape(1, 5));
            NDArray result = NDArrays.mul(multiplicand, with);
            NDArray inPlaceResult = NDArrays.muli(multiplicand, with);
            NDArray solution = manager.create(new float[] {12, 27, 48, 75, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution,
                    result,
                    "Element wise multiplication: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place multiplication: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    multiplicand,
                    inPlaceResult,
                    "Element wise multiplication: In-place operation failed");

            NDArray[] toMulAll =
                    new NDArray[] {
                        manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2)),
                        manager.create(new float[] {4, 3, 2, 1}, new Shape(2, 2)),
                        manager.create(new float[] {2, 2, 2, 2}, new Shape(2, 2))
                    };
            NDArray mulAll = NDArrays.mul(toMulAll);
            NDArray mulAllInPlace = NDArrays.muli(toMulAll);
            Assertions.assertFalse(
                    mulAll.equals(toMulAll[0]), "None in-place operator returned in-place result");
            Assertions.assertTrue(mulAllInPlace.equals(toMulAll[0]), "In-place summation failed");
            NDArray mulAllResult = manager.create(new float[] {8, 12, 12, 8}, new Shape(2, 2));
            Assertions.assertEquals(mulAllResult, mulAll, "Incorrect value in summed array");
            Assertions.assertEquals(mulAllResult, mulAllInPlace, "Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testScalarDivision() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 0}, new Shape(1, 5));
            NDArray result = NDArrays.div(dividend, 3);
            NDArray inPlaceResult = NDArrays.divi(dividend, 3);
            NDArray solution = manager.create(new float[] {2, 3, 4, 5, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    result, solution, "Scalar division: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    inPlaceResult,
                    solution,
                    "Scalar in-place division: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend, inPlaceResult, "Scalar division: In-place operation failed");
        }
    }

    @RunAsTest
    public void testElemWiseDivision() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 0}, new Shape(1, 5));
            NDArray divisor = manager.create(new float[] {2, 3, 4, 5, 6}, new Shape(1, 5));
            NDArray result = NDArrays.div(dividend, divisor);
            NDArray inPlaceResult = NDArrays.divi(dividend, divisor);
            NDArray solution = manager.create(new float[] {3, 3, 3, 3, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Element wise Division: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place division: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend, inPlaceResult, "Element wise division: In-place operation failed");
        }
    }

    @RunAsTest
    public void testReverseScalarDivision() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 45}, new Shape(1, 5));
            NDArray result = NDArrays.div(180, dividend);
            NDArray inPlaceResult = NDArrays.divi(180, dividend);
            NDArray solution = manager.create(new float[] {30, 20, 15, 12, 4}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Scalar reverse division: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place reverse division: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend, inPlaceResult, "Scalar reverse division: In-place operation failed");
        }
    }

    @RunAsTest
    public void testReverseElemWiseDivision() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 45}, new Shape(1, 5));
            NDArray divisor = manager.create(new float[] {24, 63, 96, 15, 90}, new Shape(1, 5));
            NDArray result = dividend.getNDArrayInternal().rdiv(divisor);
            NDArray inPlaceResult = dividend.getNDArrayInternal().rdivi(divisor);
            NDArray solution = manager.create(new float[] {4, 7, 8, 1, 2}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution,
                    result,
                    "Reverse Element wise Division: Incorrect value in result ndarray");
            Assertions.assertTrue(
                    NDArrays.equals(solution, inPlaceResult),
                    "Reverse Element wise in-place division: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend,
                    inPlaceResult,
                    "Reverse Element wise division: In-place operation failed");
        }
    }

    @RunAsTest
    public void testScalarRemainder() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {5, 6, 7, 8, 9}, new Shape(1, 5));
            NDArray result = NDArrays.mod(dividend, 3);
            NDArray inPlaceResult = NDArrays.modi(dividend, 3);
            NDArray solution = manager.create(new float[] {2, 0, 1, 2, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    result, solution, "Scalar Remainder: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    inPlaceResult,
                    solution,
                    "Scalar in-place Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend, inPlaceResult, "Scalar division: In-place operation failed");
        }
    }

    @RunAsTest
    public void testElemWiseRemainder() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {7, 8, 9, 10, 11}, new Shape(1, 5));
            NDArray divisor = manager.create(new float[] {2, 3, 4, 5, 6}, new Shape(1, 5));
            NDArray result = NDArrays.mod(dividend, divisor);
            NDArray inPlaceResult = NDArrays.modi(dividend, divisor);
            NDArray solution = manager.create(new float[] {1, 2, 1, 0, 5}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Element wise Remainder: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend, inPlaceResult, "Element wise Remainder: In-place operation failed");
        }
    }

    @RunAsTest
    public void testReverseScalarRemainder() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {5, 6, 7, 8, 9}, new Shape(1, 5));
            NDArray result = NDArrays.mod(180, dividend);
            NDArray inPlaceResult = NDArrays.modi(180, dividend);
            NDArray solution = manager.create(new float[] {0, 0, 5, 4, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution,
                    result,
                    "Scalar reverse Remainder: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place reverse Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend,
                    inPlaceResult,
                    "Scalar Remainder division: In-place operation failed");
        }
    }

    @RunAsTest
    public void testReverseElemWiseRemainder() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {7, 8, 9, 10, 11}, new Shape(1, 5));
            NDArray divisor = manager.create(new float[] {20, 21, 22, 23, 24}, new Shape(1, 5));
            NDArray result = dividend.getNDArrayInternal().rmod(divisor);
            NDArray inPlaceResult = dividend.getNDArrayInternal().rmodi(divisor);
            NDArray solution = manager.create(new float[] {6, 5, 4, 3, 2}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution,
                    result,
                    "Reverse Element wise Remainder: Incorrect value in result ndarray");
            Assertions.assertTrue(
                    NDArrays.equals(solution, inPlaceResult),
                    "Reverse Element wise in-place Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    dividend,
                    inPlaceResult,
                    "Reverse Element wise Remainder: In-place operation failed");
        }
    }

    @RunAsTest
    public void testPowerScalar() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {6, 0, -1, 5, 2}, new Shape(1, 5));
            NDArray result = array.pow(2);
            NDArray inPlaceResult = array.powi(2);
            NDArray solution = manager.create(new float[] {36, 0, 1, 25, 4}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Scalar power: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place power: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    array, inPlaceResult, "Scalar power: In-place operation failed");
        }
    }

    @RunAsTest
    public void testPower() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {6, 9, 12, 2, 0}, new Shape(1, 5));
            NDArray power = manager.create(new float[] {3, 0, 1, -2, 3}, new Shape(1, 5));
            NDArray result = array.pow(power);
            NDArray inPlaceResult = array.powi(power);
            NDArray solution = manager.create(new float[] {216, 1, 12, 0.25f, 0}, new Shape(1, 5));
            Assertions.assertEquals(
                    solution, result, "Scalar power: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place power: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    array, inPlaceResult, "Scalar power: In-place operation failed");
        }
    }

    @RunAsTest
    public void testReversePowerScalar() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {3, 4, 5, 6, 7});
            NDArray power = NDArrays.pow(2, array);
            NDArray inPlaceResult = NDArrays.powi(2, array);
            NDArray solution = manager.create(new float[] {8, 16, 32, 64, 128});
            Assertions.assertEquals(
                    solution, power, "Scalar reverse power: Incorrect value in result ndarray");
            Assertions.assertEquals(
                    solution,
                    inPlaceResult,
                    "Scalar in-place reverse power: Incorrect value in result ndarray");
            Assertions.assertInPlace(
                    array, inPlaceResult, "Scalar reverse division: In-place operation failed");
        }
    }
}
