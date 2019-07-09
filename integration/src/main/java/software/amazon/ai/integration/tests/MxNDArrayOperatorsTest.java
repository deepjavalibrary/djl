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

import java.util.Arrays;
import java.util.stream.DoubleStream;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDFactory;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDScopedFactory;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;

public class MxNDArrayOperatorsTest extends AbstractTest {

    NDScopedFactory factory = MxNDFactory.getSystemFactory();

    public static void main(String[] args) {
        new MxNDArrayOperatorsTest().runTest(args);
    }

    @RunAsTest
    public void testCopyTo() throws FailedTestException {
        NDArray ndArray1 = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = factory.create(new DataDesc(new Shape(1, 4)));
        ndArray1.copyTo(ndArray2);
        ndArray1.contentEquals(ndArray2);
        Assertions.assertEquals(ndArray1, ndArray2, "CopyTo NDArray failed");
    }

    @RunAsTest
    public void testEqualsForEqualNDArray() throws FailedTestException {
        NDArray ndArray1 = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray result = NDArrays.eq(ndArray1, ndArray2);
        Assertions.assertStatement(
                result.nonzero() == 4 && NDArrays.equals(ndArray1, ndArray2),
                "Incorrect comparison for equal NDArray");
    }

    @RunAsTest
    public void testEqualsForScalar() throws FailedTestException {
        NDArray ndArray = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray result = NDArrays.eq(ndArray, 2);
        Assertions.assertStatement(result.nonzero() == 1, "Incorrect comparison for equal NDArray");
    }

    @RunAsTest
    public void testEqualsForUnEqualNDArray() throws FailedTestException {
        NDArray ndArray1 = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = factory.create(new Shape(1, 4), new float[] {1f, 3f, 3f, 4f});
        NDArray result = NDArrays.eq(ndArray1, ndArray2);
        Assertions.assertStatement(
                result.nonzero() == 3 && !NDArrays.equals(ndArray1, ndArray2),
                "Incorrect comparison for unequal NDArray");
    }

    @RunAsTest
    public void testNonZero() throws FailedTestException {
        NDArray ndArray1 = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray ndArray2 = factory.create(new Shape(1, 4), new float[] {1f, 2f, 0f, 4f});
        NDArray ndArray3 = factory.create(new Shape(1, 4), new float[] {0f, 0f, 0f, 4f});
        NDArray ndArray4 = factory.create(new Shape(1, 4), new float[] {0f, 0f, 0f, 0f});
        Assertions.assertStatement(
                ndArray1.nonzero() == 4
                        && ndArray2.nonzero() == 3
                        && ndArray3.nonzero() == 1
                        && ndArray4.nonzero() == 0,
                "nonzero function returned incorrect value");
    }

    @RunAsTest
    public void testAddScalar() throws FailedTestException {
        NDArray addend = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray result = NDArrays.add(addend, 2);
        Assertions.assertStatement(!NDArrays.equals(addend, result), "In-place summation failed");
        NDArray solution = factory.create(new Shape(1, 4), new float[] {3f, 4f, 5f, 6f});
        Assertions.assertEquals(solution, result, "Incorrect value in summed array");
    }

    @RunAsTest
    public void testAddScalarInPlace() throws FailedTestException {
        MxNDArray addend =
                (MxNDArray) factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        MxNDArray result = (MxNDArray) NDArrays.addi(addend, 2);
        Assertions.assertInPlace(result, addend, "In-place summation failed");
        NDArray solution = factory.create(new Shape(1, 4), new float[] {3f, 4f, 5f, 6f});
        Assertions.assertEquals(solution, result, "Incorrect value in summed array");
    }

    @RunAsTest
    public void testAddNDArray() throws FailedTestException {
        NDArray addend = factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        NDArray addendum = factory.create(new Shape(1, 4), new float[] {2f, 3f, 4f, 5f});
        NDArray result = NDArrays.add(addend, addendum);
        Assertions.assertStatement(!NDArrays.equals(addend, result), "In-place summation failed");
        NDArray solution = factory.create(new Shape(1, 4), new float[] {3f, 5f, 7f, 9f});
        Assertions.assertEquals(solution, result, "Incorrect value in summed array");

        NDArray[] toAddAll =
                new NDArray[] {
                    factory.create(new Shape(2, 2), new float[] {1, 2, 3, 4}),
                    factory.create(new Shape(2, 2), new float[] {4, 3, 2, 1}),
                    factory.create(new Shape(2, 2), new float[] {2, 2, 2, 2})
                };
        NDArray addAll = NDArrays.add(toAddAll);
        Assertions.assertStatement(!addAll.equals(toAddAll[0]), "In-place summation failed");
        NDArray addAllResult = factory.create(new Shape(2, 2), new float[] {7, 7, 7, 7});
        Assertions.assertEquals(addAllResult, addAll, "Incorrect value in summed array");
    }

    @RunAsTest
    public void testAddNDArrayInPlace() throws FailedTestException {
        MxNDArray addend =
                (MxNDArray) factory.create(new Shape(1, 4), new float[] {1f, 2f, 3f, 4f});
        MxNDArray addendum =
                (MxNDArray) factory.create(new Shape(1, 4), new float[] {2f, 3f, 4f, 5f});
        MxNDArray result = (MxNDArray) NDArrays.addi(addend, addendum);
        Assertions.assertInPlace(result, addend, "In-place summation failed");
        NDArray solution = factory.create(new Shape(1, 4), new float[] {3f, 5f, 7f, 9f});
        Assertions.assertEquals(solution, result, "Incorrect value in summed array");

        NDArray[] toAddAll =
                new NDArray[] {
                    factory.create(new Shape(2, 2), new float[] {1, 2, 3, 4}),
                    factory.create(new Shape(2, 2), new float[] {4, 3, 2, 1}),
                    factory.create(new Shape(2, 2), new float[] {2, 2, 2, 2})
                };
        NDArray addAll = NDArrays.addi(toAddAll);
        Assertions.assertStatement(addAll.equals(toAddAll[0]), "In-place summation failed");
        NDArray addAllResult = factory.create(new Shape(2, 2), new float[] {7, 7, 7, 7});
        Assertions.assertEquals(addAllResult, addAll, "Incorrect value in summed array");
    }

    @RunAsTest
    public void testGreaterThanScalar() throws FailedTestException {
        NDArray array = factory.create(new Shape(1, 5), new float[] {1, 0, 2f, 2f, 4f});
        NDArray greater = NDArrays.gt(array, 2);
        Assertions.assertStatement(greater.nonzero() == 1, "greater_scalar: Incorrect comparison");
    }

    @RunAsTest
    public void testGreaterThanOrEqualToScalar() throws FailedTestException {
        NDArray array = factory.create(new Shape(1, 4), new float[] {1f, 2f, 2f, 4f});
        NDArray greater = NDArrays.gte(array, 2);
        Assertions.assertStatement(
                greater.nonzero() == 3, "greater_equals_scalar: Incorrect comparison");
    }

    @RunAsTest
    public void testLesserThanOrEqualToScalar() throws FailedTestException {
        NDArray array = factory.create(new Shape(1, 5), new float[] {1f, 2f, 2f, 4f, 5f});
        NDArray greater = NDArrays.lte(array, 2);
        Assertions.assertStatement(
                greater.nonzero() == 3, "lesser_equals_scalar: Incorrect comparison");
    }

    @RunAsTest
    public void testLesserThanScalar() throws FailedTestException {
        NDArray array = factory.create(new Shape(1, 5), new float[] {1f, 2f, 2f, 4f, 5f});
        NDArray greater = NDArrays.lt(array, 2);
        Assertions.assertStatement(greater.nonzero() == 1, "lesser_scalar: Incorrect comparison");
    }

    @RunAsTest
    public void testGreaterThanAndLessThan() throws FailedTestException {
        NDArray ndArray1 = factory.create(new Shape(1, 6), new float[] {1f, 2f, 2f, 4f, 5f, 4f});
        NDArray ndArray2 = factory.create(new Shape(1, 6), new float[] {2f, 1f, 2f, 5f, 4f, 5f});
        NDArray greater = NDArrays.gt(ndArray1, ndArray2);
        Assertions.assertStatement(greater.nonzero() == 2, "greater: Incorrect comparison");
        NDArray lesser = NDArrays.lt(ndArray1, ndArray2);
        Assertions.assertStatement(lesser.nonzero() == 3, "lesser: Incorrect comparison");
    }

    @RunAsTest
    public void testGreaterThanAndLessThanEquals() throws FailedTestException {
        NDArray ndArray1 = factory.create(new Shape(1, 6), new float[] {1f, 2f, 2f, 4f, 5f, 4f});
        NDArray ndArray2 = factory.create(new Shape(1, 6), new float[] {2f, 1f, 2f, 5f, 4f, 5f});
        NDArray greater = NDArrays.gte(ndArray1, ndArray2);
        Assertions.assertStatement(greater.nonzero() == 3, "greater_equal: Incorrect comparison");
        NDArray lesser = NDArrays.lte(ndArray1, ndArray2);
        Assertions.assertStatement(lesser.nonzero() == 4, "lesser_equal: Incorrect comparison");
    }

    @RunAsTest
    public void testArange() throws FailedTestException {
        NDArray expectedND =
                factory.create(new Shape(10), new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
        NDArray testedND = factory.arange(0, 10, 1);
        Assertions.assertEquals(testedND, expectedND);
        testedND = factory.arange(0, 10, 1);
        Assertions.assertEquals(testedND, expectedND);
        testedND = factory.arange(10);
        Assertions.assertEquals(testedND, expectedND);
    }

    @RunAsTest
    public void testLinspace() throws FailedTestException {
        NDArray expectedND =
                factory.create(new Shape(10), new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
        NDArray testedND = factory.linspace(0.0, 9.0, 10, true, null);
        Assertions.assertEquals(testedND, expectedND);
    }

    @RunAsTest
    public void testCumsum() throws FailedTestException {
        NDArray expectedND =
                factory.create(new Shape(10), new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
        NDArray actualND =
                factory.create(
                        new Shape(10), new float[] {0f, 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f});
        Assertions.assertEquals(expectedND.cumsum(0), actualND);
    }

    @RunAsTest
    public void testCumsumi() throws FailedTestException {
        MxNDArray expectedND =
                (MxNDArray)
                        factory.create(
                                new Shape(10),
                                new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
        MxNDArray actualND =
                (MxNDArray)
                        factory.create(
                                new Shape(10),
                                new float[] {0f, 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f});
        Assertions.assertEquals(expectedND.cumsumi(0), actualND);
        Assertions.assertInPlace((MxNDArray) expectedND.cumsumi(0), expectedND);
    }

    @RunAsTest
    public void testTile() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});

        NDArray tileAll = original.tile(2);
        NDArray tileAllExpected =
                factory.create(
                        new Shape(4, 4),
                        new float[] {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4});
        Assertions.assertEquals(tileAll, tileAllExpected, "Incorrect tile all");

        NDArray tileAxis = original.tile(0, 3);
        NDArray tileAxisExpected =
                factory.create(new Shape(6, 2), new float[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
        Assertions.assertEquals(tileAxis, tileAxisExpected, "Incorrect tile on axis");

        NDArray tileArray = original.tile(new long[] {3, 1});
        Assertions.assertStatement(
                tileArray.contentEquals(tileAxisExpected), "Incorrect tile array");

        NDArray tileShape = original.tile(new Shape(4));
        NDArray tileShapeExpected =
                factory.create(new Shape(2, 4), new float[] {1, 2, 1, 2, 3, 4, 3, 4});
        Assertions.assertEquals(tileShape, tileShapeExpected, "Incorrect tile shape");
    }

    @RunAsTest
    public void testRepeat() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});

        NDArray repeatAll = original.repeat(2);
        NDArray repeatAllExpected =
                factory.create(
                        new Shape(4, 4),
                        new float[] {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
        Assertions.assertEquals(repeatAll, repeatAllExpected, "Incorrect repeat all");

        NDArray repeatAxis = original.repeat(0, 3);
        NDArray repeatAxisExpected =
                factory.create(new Shape(6, 2), new float[] {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4});
        Assertions.assertEquals(repeatAxis, repeatAxisExpected, "Incorrect repeat on axis");

        NDArray repeatArray = original.repeat(new long[] {3, 1});
        Assertions.assertEquals(repeatArray, repeatAxisExpected, "Incorrect repeat array");

        NDArray repeatShape = original.repeat(new Shape(4));
        NDArray repeatShapeExpected =
                factory.create(new Shape(2, 4), new float[] {1, 1, 2, 2, 3, 3, 4, 4});
        Assertions.assertEquals(repeatShape, repeatShapeExpected, "Incorrect repeat shape");
    }

    @RunAsTest
    public void testScalarDivision() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 15, 0});
        NDArray result = NDArrays.div(dividend, 3);
        NDArray inPlaceResult = NDArrays.divi(dividend, 3);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {2, 3, 4, 5, 0});
        Assertions.assertEquals(
                result, solution, "Scalar division: Incorrect value in result ndarray");
        Assertions.assertEquals(
                inPlaceResult,
                solution,
                "Scalar in-place division: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Scalar division: In-place operation failed");
    }

    @RunAsTest
    public void testElemWiseDivision() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 15, 0});
        NDArray divisor = factory.create(new Shape(1, 5), new float[] {2, 3, 4, 5, 6});
        NDArray result = NDArrays.div(dividend, divisor);
        NDArray inPlaceResult = NDArrays.divi(dividend, divisor);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {3, 3, 3, 3, 0});
        Assertions.assertEquals(
                solution, result, "Element wise Division: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place division: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Element wise division: In-place operation failed");
    }

    @RunAsTest
    public void testReverseScalarDivision() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 15, 45});
        NDArray result = NDArrays.div(180, dividend);
        NDArray inPlaceResult = NDArrays.divi(180, dividend);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {30, 20, 15, 12, 4});
        Assertions.assertEquals(
                solution, result, "Scalar reverse division: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place reverse division: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Scalar reverse division: In-place operation failed");
    }

    @RunAsTest
    public void testReverseElemWiseDivision() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 15, 45});
        NDArray divisor = factory.create(new Shape(1, 5), new float[] {24, 63, 96, 15, 90});
        NDArray result = dividend.getNDArrayInternal().rdiv(divisor);
        NDArray inPlaceResult = dividend.getNDArrayInternal().rdivi(divisor);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {4, 7, 8, 1, 2});
        Assertions.assertEquals(
                solution,
                result,
                "Reverse Element wise Division: Incorrect value in result ndarray");
        Assertions.assertStatement(
                NDArrays.equals(solution, inPlaceResult),
                "Reverse Element wise in-place division: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Reverse Element wise division: In-place operation failed");
    }

    @RunAsTest
    public void testScalarSubtraction() throws FailedTestException {
        NDArray minuend = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 11, 0});
        NDArray result = NDArrays.sub(minuend, 3);
        NDArray inPlaceResult = NDArrays.subi(minuend, 3);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {3, 6, 9, 8, -3});
        Assertions.assertEquals(
                solution, result, "Scalar subtraction: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place subtraction: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) minuend,
                (MxNDArray) inPlaceResult,
                "Scalar subtraction: In-place operation failed");
    }

    @RunAsTest
    public void testElemWiseSubtraction() throws FailedTestException {
        NDArray minuend = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 15, 0});
        NDArray subtrahend = factory.create(new Shape(1, 5), new float[] {2, 3, 4, 5, 6});
        NDArray result = NDArrays.sub(minuend, subtrahend);
        NDArray inPlaceResult = NDArrays.subi(minuend, subtrahend);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {4, 6, 8, 10, -6});
        Assertions.assertEquals(
                solution, result, "Element wise subtraction: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place subtraction: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) minuend,
                (MxNDArray) inPlaceResult,
                "Element wise subtraction: In-place operation failed");
    }

    @RunAsTest
    public void testReverseScalarSubtraction() throws FailedTestException {
        NDArray minuend = factory.create(new Shape(1, 5), new float[] {6, 91, 12, 215, 180});
        NDArray result = NDArrays.sub(180, minuend);
        NDArray inPlaceResult = NDArrays.subi(180, minuend);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {174, 89, 168, -35, 0});
        Assertions.assertEquals(
                solution, result, "Scalar reverse subtraction: Incorrect value in result ndarray");
        Assertions.assertStatement(
                NDArrays.equals(solution, inPlaceResult),
                "Scalar in-place reverse subtraction: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) minuend,
                (MxNDArray) inPlaceResult,
                "Scalar reverse subtraction: In-place operation failed");
    }

    @RunAsTest
    public void testReverseElemWiseSubtraction() throws FailedTestException {
        NDArray minuend = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 15, 45});
        NDArray subtrahend = factory.create(new Shape(1, 5), new float[] {24, 63, 96, 15, 90});
        NDArray result = minuend.getNDArrayInternal().rsub(subtrahend);
        NDArray inPlaceResult = minuend.getNDArrayInternal().rsubi(subtrahend);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {18, 54, 84, 0, 45});
        Assertions.assertEquals(
                solution,
                result,
                "Reverse Element wise subtraction: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Reverse Element wise in-place subtraction: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) minuend,
                (MxNDArray) inPlaceResult,
                "Reverse Element wise subtraction: In-place operation failed");
    }

    @RunAsTest
    public void testScalarMultiplication() throws FailedTestException {
        NDArray multiplicand = factory.create(new Shape(1, 5), new float[] {6, 9, -12, 15, 0});
        NDArray result = NDArrays.mul(multiplicand, 3);
        NDArray inPlaceResult = NDArrays.muli(multiplicand, 3);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {18, 27, -36, 45, 0});
        Assertions.assertEquals(
                solution, result, "Scalar multiplication: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place multiplication: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) multiplicand,
                (MxNDArray) inPlaceResult,
                "Scalar multiplication: In-place operation failed");
    }

    @RunAsTest
    public void testElemWiseMultiplication() throws FailedTestException {
        NDArray multiplicand = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 15, 0});
        NDArray with = factory.create(new Shape(1, 5), new float[] {2, 3, 4, 5, 6});
        NDArray result = NDArrays.mul(multiplicand, with);
        NDArray inPlaceResult = NDArrays.muli(multiplicand, with);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {12, 27, 48, 75, 0});
        Assertions.assertEquals(
                solution, result, "Element wise multiplication: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place multiplication: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) multiplicand,
                (MxNDArray) inPlaceResult,
                "Element wise multiplication: In-place operation failed");

        NDArray[] toMulAll =
                new NDArray[] {
                    factory.create(new Shape(2, 2), new float[] {1, 2, 3, 4}),
                    factory.create(new Shape(2, 2), new float[] {4, 3, 2, 1}),
                    factory.create(new Shape(2, 2), new float[] {2, 2, 2, 2})
                };
        NDArray mulAll = NDArrays.mul(toMulAll);
        NDArray mulAllInPlace = NDArrays.muli(toMulAll);
        Assertions.assertStatement(!mulAll.equals(toMulAll[0]), "In-place summation failed");
        Assertions.assertStatement(mulAllInPlace.equals(toMulAll[0]), "In-place summation failed");
        NDArray mulAllResult = factory.create(new Shape(2, 2), new float[] {8, 12, 12, 8});
        Assertions.assertEquals(mulAllResult, mulAll, "Incorrect value in summed array");
        Assertions.assertEquals(mulAllResult, mulAllInPlace, "Incorrect value in summed array");
    }

    @RunAsTest
    public void testScalarRemainder() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {5, 6, 7, 8, 9});
        NDArray result = NDArrays.mod(dividend, 3);
        NDArray inPlaceResult = NDArrays.modi(dividend, 3);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {2, 0, 1, 2, 0});
        Assertions.assertEquals(
                result, solution, "Scalar Remainder: Incorrect value in result ndarray");
        Assertions.assertEquals(
                inPlaceResult,
                solution,
                "Scalar in-place Remainder: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Scalar division: In-place operation failed");
    }

    @RunAsTest
    public void testElemWiseRemainder() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {7, 8, 9, 10, 11});
        NDArray divisor = factory.create(new Shape(1, 5), new float[] {2, 3, 4, 5, 6});
        NDArray result = NDArrays.mod(dividend, divisor);
        NDArray inPlaceResult = NDArrays.modi(dividend, divisor);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {1, 2, 1, 0, 5});
        Assertions.assertEquals(
                solution, result, "Element wise Remainder: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place Remainder: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Element wise Remainder: In-place operation failed");
    }

    @RunAsTest
    public void testReverseScalarRemainder() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {5, 6, 7, 8, 9});
        NDArray result = NDArrays.mod(180, dividend);
        NDArray inPlaceResult = NDArrays.modi(180, dividend);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {0, 0, 5, 4, 0});
        Assertions.assertEquals(
                solution, result, "Scalar reverse Remainder: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place reverse Remainder: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Scalar Remainder division: In-place operation failed");
    }

    @RunAsTest
    public void testReverseElemWiseRemainder() throws FailedTestException {
        NDArray dividend = factory.create(new Shape(1, 5), new float[] {7, 8, 9, 10, 11});
        NDArray divisor = factory.create(new Shape(1, 5), new float[] {20, 21, 22, 23, 24});
        NDArray result = dividend.getNDArrayInternal().rmod(divisor);
        NDArray inPlaceResult = dividend.getNDArrayInternal().rmodi(divisor);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {6, 5, 4, 3, 2});
        Assertions.assertEquals(
                solution,
                result,
                "Reverse Element wise Remainder: Incorrect value in result ndarray");
        Assertions.assertStatement(
                NDArrays.equals(solution, inPlaceResult),
                "Reverse Element wise in-place Remainder: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) dividend,
                (MxNDArray) inPlaceResult,
                "Reverse Element wise Remainder: In-place operation failed");
    }

    @RunAsTest
    public void testNegation() throws FailedTestException {
        NDArray ndArray = factory.create(new Shape(1, 5), new float[] {6, 9, -12, -11, 0});
        NDArray result = ndArray.neg();
        NDArray inPlaceResult = ndArray.negi();
        NDArray solution = factory.create(new Shape(1, 5), new float[] {-6, -9, 12, 11, 0});
        Assertions.assertEquals(
                solution, result, "Scalar subtraction: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place subtraction: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) ndArray,
                (MxNDArray) inPlaceResult,
                "Scalar subtraction: In-place operation failed");
    }

    @RunAsTest
    public void testMax() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {2, 4, 6, 8});

        Float maxAll = (Float) original.max();
        Assertions.assertEquals(8, maxAll, "Incorrect max all");

        NDArray maxAxes = original.max(new int[] {1});
        NDArray maxAxesExpected = factory.create(new Shape(2), new float[] {4, 8});
        Assertions.assertEquals(maxAxesExpected, maxAxes, "Incorrect max axes");

        NDArray maxKeep = original.max(new int[] {0}, true);
        NDArray maxKeepExpected = factory.create(new Shape(1, 2), new float[] {6, 8});
        Assertions.assertEquals(maxKeepExpected, maxKeep, "Incorrect max keep");
    }

    @RunAsTest
    public void testMin() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {2, 4, 6, 8});

        Float minAll = (Float) original.min();
        Assertions.assertEquals(2, minAll, "Incorrect min all");

        NDArray minAxes = original.min(new int[] {1});
        NDArray minAxesExpected = factory.create(new Shape(2), new float[] {2, 6});
        Assertions.assertEquals(minAxesExpected, minAxes, "Incorrect min axes");

        NDArray minKeep = original.min(new int[] {0}, true);
        NDArray minKeepExpected = factory.create(new Shape(1, 2), new float[] {2, 4});
        Assertions.assertEquals(minKeepExpected, minKeep, "Incorrect min keep");
    }

    @RunAsTest
    public void testSum() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {2, 4, 6, 8});

        Float sumAll = (Float) original.sum();
        Assertions.assertEquals(20, sumAll, "Incorrect sum all");

        NDArray sumAxes = original.sum(new int[] {1});
        NDArray sumAxesExpected = factory.create(new Shape(2), new float[] {6, 14});
        Assertions.assertEquals(sumAxesExpected, sumAxes, "Incorrect sum axes");

        NDArray sumKeep = original.sum(new int[] {0}, true);
        NDArray sumKeepExpected = factory.create(new Shape(1, 2), new float[] {8, 12});
        Assertions.assertEquals(sumKeepExpected, sumKeep, "Incorrect sum keep");
    }

    @RunAsTest
    public void testProd() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {2, 4, 6, 8});

        Float prodAll = (Float) original.prod();
        Assertions.assertEquals(384, prodAll, "Incorrect max axes");
        if (prodAll != 384) {
            throw new FailedTestException("Incorrect prod all");
        }

        NDArray prodAxes = original.prod(new int[] {1});
        NDArray prodAxesExpected = factory.create(new Shape(2), new float[] {8, 48});
        Assertions.assertEquals(prodAxesExpected, prodAxes, "Incorrect prod axes");

        NDArray prodKeep = original.prod(new int[] {0}, true);
        NDArray prodKeepExpected = factory.create(new Shape(1, 2), new float[] {12, 32});
        Assertions.assertEquals(prodKeepExpected, prodKeep, "Incorrect prod keep");
    }

    @RunAsTest
    public void testMean() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {2, 4, 6, 8});

        Float meanAll = (Float) original.mean();
        Assertions.assertEquals(5, meanAll, "Incorrect mean all");

        NDArray meanAxes = original.mean(new int[] {1});
        NDArray meanAxesExpected = factory.create(new Shape(2), new float[] {3, 7});
        Assertions.assertEquals(meanAxesExpected, meanAxes, "Incorrect mean axes");

        NDArray meanKeep = original.mean(new int[] {0}, true);
        NDArray meanKeepExpected = factory.create(new Shape(1, 2), new float[] {4, 6});
        Assertions.assertEquals(meanKeepExpected, meanKeep, "Incorrect mean keep");
    }

    @RunAsTest
    public void testLogicalNot() throws FailedTestException {
        double[] testedData = new double[] {-2., 0., 1.};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        double[] boolData = new double[] {0.0, 1.0, 0.0};
        NDArray expectedND = factory.create(new Shape(testedData.length), boolData);
        Assertions.assertAlmostEquals(testedND.logicalNot(), expectedND);
    }

    @RunAsTest
    public void testAbs() throws FailedTestException {
        double[] testedData = new double[] {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::abs).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.abs(), expectedND);
    }

    @RunAsTest
    public void testSquare() throws FailedTestException {
        double[] testedData = new double[] {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(x -> Math.pow(x, 2.0)).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.square(), expectedND);
    }

    @RunAsTest
    public void testCbrt() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::cbrt).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.cbrt(), expectedND);
    }

    @RunAsTest
    public void testFloor() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::floor).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.floor(), expectedND);
    }

    @RunAsTest
    public void testCeil() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::ceil).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.ceil(), expectedND);
    }

    @RunAsTest
    public void testRound() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::round).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.round(), expectedND);
    }

    @RunAsTest
    public void testTrunc() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        double[] truncData = new double[] {1.0, 2.0, -3, -4, 5, -223};
        NDArray expectedND = factory.create(new Shape(testedData.length), truncData);
        Assertions.assertAlmostEquals(testedND.trunc(), expectedND);
    }

    @RunAsTest
    public void testExp() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::exp).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.exp(), expectedND);
    }

    @RunAsTest
    public void testLog() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::log).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.log(), expectedND);
    }

    @RunAsTest
    public void testLog10() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::log10).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.log10(), expectedND);
    }

    @RunAsTest
    public void testLog2() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(x -> Math.log10(x) / Math.log10(2)).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.log2(), expectedND);
    }

    @RunAsTest
    public void testSin() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::sin).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.sin(), expectedND);
    }

    @RunAsTest
    public void testCos() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::cos).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.cos(), expectedND);
    }

    @RunAsTest
    public void testTan() throws FailedTestException {
        double[] testedData = new double[] {0.0, Math.PI / 4.0, Math.PI / 2.0};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::tan).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.tan(), expectedND);
    }

    @RunAsTest
    public void testAsin() throws FailedTestException {
        double[] testedData = new double[] {1.0, -1.0, -0.22, 0.4, 0.1234};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::asin).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.asin(), expectedND);
    }

    @RunAsTest
    public void testAcos() throws FailedTestException {
        double[] testedData = new double[] {-1.0, -0.707, 0.0, 0.707, 1.0};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::acos).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.acos(), expectedND);
    }

    @RunAsTest
    public void testAtan() throws FailedTestException {
        double[] testedData = new double[] {-1.0, 0.0, 1.0};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::atan).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.atan(), expectedND);
    }

    @RunAsTest
    public void testToDegrees() throws FailedTestException {
        double[] testedData = new double[] {0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::toDegrees).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.toDegrees(), expectedND);
    }

    @RunAsTest
    public void testToRadians() throws FailedTestException {
        double[] testedData = new double[] {0.0, 90.0, 180.0, 270.0, 360.0};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::toRadians).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.toRadians(), expectedND);
    }

    @RunAsTest
    public void testSinh() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::sinh).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.sinh(), expectedND);
    }

    @RunAsTest
    public void testCosh() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::cosh).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.cosh(), expectedND);
    }

    @RunAsTest
    public void testTanh() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        testedData = DoubleStream.of(testedData).map(Math::tanh).toArray();
        NDArray expectedND = factory.create(new Shape(testedData.length), testedData);
        Assertions.assertAlmostEquals(testedND.tanh(), expectedND);
    }

    @RunAsTest
    public void testAsinh() throws FailedTestException {
        double[] testedData = new double[] {Math.E, 10.0};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        double[] aSinhData = new double[] {1.72538256, 2.99822295};
        NDArray expectedND = factory.create(new Shape(testedData.length), aSinhData);
        Assertions.assertAlmostEquals(testedND.asinh(), expectedND);
    }

    @RunAsTest
    public void testAtanh() throws FailedTestException {
        double[] testedData = new double[] {0.0, -0.5};
        NDArray testedND = factory.create(new Shape(testedData.length), testedData);
        double[] aTanhData = new double[] {0.0, -0.54930614};
        NDArray expectedND = factory.create(new Shape(testedData.length), aTanhData);
        Assertions.assertAlmostEquals(testedND.atanh(), expectedND);
    }

    @RunAsTest
    public void testExpandDim() throws FailedTestException {
        NDArray original = factory.create(new Shape(2), new int[] {1, 2});
        Assertions.assertStatement(
                Arrays.equals(
                        original.expandDims(0).getShape().getShape(), new Shape(1, 2).getShape()));
    }

    @RunAsTest
    public void testStack() throws FailedTestException {
        NDArray stackedND = factory.create(new Shape(1), new float[] {1f});
        NDArray stackedND2 = factory.create(new Shape(1), new float[] {2f});
        NDArray actual = factory.create(new Shape(2, 1), new float[] {1f, 2f});

        Assertions.assertEquals(stackedND.stack(new NDArray[] {stackedND2}, 0), actual);
        Assertions.assertEquals(stackedND.stack(stackedND2, 0), actual);
        Assertions.assertEquals(NDArrays.stack(new NDArray[] {stackedND, stackedND2}, 0), actual);
    }

    @RunAsTest
    public void testConcat() throws FailedTestException {
        NDArray concatedND = factory.create(new Shape(1), new float[] {1f});
        NDArray concatedND2 = factory.create(new Shape(1), new float[] {2f});
        NDArray actual = factory.create(new Shape(2), new float[] {1f, 2f});

        Assertions.assertEquals(concatedND.concat(new NDArray[] {concatedND2}, 0), actual);
        Assertions.assertEquals(
                NDArrays.concat(new NDArray[] {concatedND, concatedND2}, 0), actual);
        Assertions.assertEquals(concatedND.concat(concatedND2), actual);
    }

    @RunAsTest
    public void testClip() throws FailedTestException {
        NDArray original = factory.create(new Shape(4), new float[] {1f, 2f, 3f, 4f});
        NDArray actual = factory.create(new Shape(4), new float[] {2f, 2f, 3f, 3f});

        Assertions.assertEquals(original.clip(2.0, 3.0), actual);
    }

    @RunAsTest
    public void testReshape() throws FailedTestException {
        NDArray original = factory.create(new Shape(3, 2), new float[] {1f, 2f, 3f, 4f, 5f, 6f});
        NDArray reshaped = original.reshape(new Shape(2, 3));
        NDArray expected = factory.create(new Shape(2, 3), new float[] {1f, 2f, 3f, 4f, 5f, 6f});
        Assertions.assertEquals(reshaped, expected);
    }

    @RunAsTest
    public void testFlatten() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});
        NDArray flattened = original.flatten();
        NDArray expected = factory.create(new Shape(4), new float[] {1f, 2f, 3f, 4f});
        Assertions.assertEquals(flattened, expected);
    }

    @RunAsTest
    public void testGet() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});
        Assertions.assertEquals(original.get(new NDIndex()), original);

        NDArray getAt = original.get(0);
        NDArray getAtExpected = factory.create(new Shape(2), new float[] {1f, 2f});
        Assertions.assertEquals(getAt, getAtExpected);

        NDArray getSlice = original.get("1:");
        NDArray getSliceExpected = factory.create(new Shape(1, 2), new float[] {3f, 4f});
        Assertions.assertEquals(getSlice, getSliceExpected);
    }

    @RunAsTest
    public void testSort() throws FailedTestException {
        NDArray original = factory.create(new Shape(2, 2), new float[] {2f, 1f, 4f, 3f});
        NDArray expected = factory.create(new Shape(2, 2), new float[] {1f, 2f, 3f, 4f});
        Assertions.assertEquals(original.sort(), expected);
    }

    @RunAsTest
    public void testTranspose() throws FailedTestException {
        NDArray original = factory.create(new Shape(1, 2, 2), new float[] {1f, 2f, 3f, 4f});

        NDArray transposeAll = original.transpose();
        NDArray transposeAllExpected = factory.create(new Shape(2, 2, 1), new float[] {1, 3, 2, 4});
        Assertions.assertEquals(transposeAll, transposeAllExpected, "Incorrect transpose all");

        NDArray transpose = original.transpose(new int[] {1, 0, 2});
        NDArray transposeExpected = factory.create(new Shape(2, 1, 2), new float[] {1, 2, 3, 4});
        Assertions.assertEquals(transpose, transposeExpected, "Incorrect transpose all");
        Assertions.assertEquals(original.swapAxes(0, 1), transposeExpected, "Incorrect swap axes");
    }

    @RunAsTest
    public void testPower() throws FailedTestException {
        NDArray array = factory.create(new Shape(1, 5), new float[] {6, 9, 12, 2, 0});
        NDArray power = factory.create(new Shape(1, 5), new float[] {3, 0, 1, -2, 3});
        NDArray result = array.pow(power);
        NDArray inPlaceResult = array.powi(power);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {216, 1, 12, 0.25f, 0});
        Assertions.assertEquals(
                solution, result, "Scalar power: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place power: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) array,
                (MxNDArray) inPlaceResult,
                "Scalar power: In-place operation failed");
    }

    @RunAsTest
    public void testPowerScalar() throws FailedTestException {
        NDArray array = factory.create(new Shape(1, 5), new float[] {6, 0, -1, 5, 2});
        NDArray result = array.pow(2);
        NDArray inPlaceResult = array.powi(2);
        NDArray solution = factory.create(new Shape(1, 5), new float[] {36, 0, 1, 25, 4});
        Assertions.assertEquals(
                solution, result, "Scalar power: Incorrect value in result ndarray");
        Assertions.assertEquals(
                solution,
                inPlaceResult,
                "Scalar in-place power: Incorrect value in result ndarray");
        Assertions.assertInPlace(
                (MxNDArray) array,
                (MxNDArray) inPlaceResult,
                "Scalar power: In-place operation failed");
    }

    @RunAsTest
    public void testArgMax() throws FailedTestException {
        NDArray original =
                factory.create(
                        new Shape(4, 5),
                        new float[] {
                            1, 2, 3, 4, 4, 5, 6, 23, 54, 234, 54, 23, 54, 4, 34, 34, 23, 54, 4, 3
                        });
        NDArray argMax = original.argMax();
        NDArray expected = factory.create(new Shape(1), new float[] {9});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMax(0, true);
        expected = factory.create(new Shape(1, 5), new float[] {2, 2, 2, 1, 1});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMax(1, false);
        expected = factory.create(new Shape(4), new float[] {3, 4, 0, 2});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");
    }

    @RunAsTest
    public void testArgMin() throws FailedTestException {
        NDArray original =
                factory.create(
                        new Shape(4, 5),
                        new float[] {
                            1, 23, 3, 74, 4, 5, 6, -23, -54, 234, 54, 2, 54, 4, -34, 34, 23, -54, 4,
                            3
                        });
        NDArray argMax = original.argMin();
        NDArray expected = factory.create(new Shape(1), new float[] {8});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMin(0, false);
        expected = factory.create(new Shape(5), new float[] {0, 2, 3, 1, 2});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");

        argMax = original.argMin(1, true);
        expected = factory.create(new Shape(4, 1), new float[] {0, 3, 4, 2});
        Assertions.assertEquals(argMax, expected, "Argmax: Incorrect value");
    }

    public void testMatrixMultiplication() throws FailedTestException {
        NDArray multiplicand = factory.create(new Shape(2, 3), new float[] {6, -9, -12, 15, 0, 4});
        NDArray multiplier = factory.create(new Shape(3, 1), new float[] {2, 3, -4});
        NDArray result = NDArrays.mmul(multiplicand, multiplier);
        NDArray solution = factory.create(new Shape(2, 1), new float[] {33, 14});
        Assertions.assertEquals(
                solution, result, "Matrix multiplication: Incorrect value in result ndarray");
    }
}
