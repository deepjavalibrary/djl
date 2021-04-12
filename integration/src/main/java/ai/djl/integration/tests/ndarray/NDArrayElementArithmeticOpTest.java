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

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import java.util.function.BiFunction;
import java.util.function.Function;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayElementArithmeticOpTest {

    private void testCornerCase(
            NDManager manager,
            BiFunction<NDArray, NDArray, NDArray> arrayArithmeticOp,
            BiFunction<Float, Float, Float> arithmeticOp,
            boolean inplace) {
        // test scalar with scalar
        float x1 = 10f;
        float x2 = 2f;
        NDArray array1 = manager.create(x1);
        NDArray array2 = manager.create(x2);
        NDArray result = arrayArithmeticOp.apply(array1, array2);
        Assertions.assertAlmostEquals(result.getFloat(), arithmeticOp.apply(x1, x2));
        if (inplace) {
            Assert.assertSame(array1, result);
        } else {
            // other cases only apply to non inplace test
            // test NDArray with scalar
            x1 = 10f;
            x2 = 5f;
            float y = arithmeticOp.apply(x1, x2);
            array1 = manager.create(new float[] {x1, x1});
            array2 = manager.create(x2);
            NDArray expected = manager.create(new float[] {y, y});
            Assert.assertEquals(arrayArithmeticOp.apply(array1, array2), expected);

            // test zero-dim with zero-dim
            array1 = manager.create(new Shape(4, 0, 1));
            array2 = manager.create(new Shape(1, 0));
            expected = manager.create(new Shape(4, 0, 0));
            Assert.assertEquals(arrayArithmeticOp.apply(array1, array2), expected);

            // test NDArray with zero-dim
            array1 = manager.create(new float[] {10f});
            array2 = manager.create(new Shape(2, 0, 3));
            expected = manager.create(new Shape(2, 0, 3));
            Assert.assertEquals(arrayArithmeticOp.apply(array1, array2), expected);
        }
    }

    private void testReverseCornerCase(
            NDManager manager,
            NDArray scalarNDArray,
            Function<NDArray, NDArray> arrayArithmeticOp,
            BiFunction<Float, Float, Float> arithmeticOp,
            boolean inplace) {
        // scalar with number
        float x1 = scalarNDArray.getFloat();
        float x2 = 2f;
        NDArray ndArray2 = manager.create(x2);
        NDArray result = arrayArithmeticOp.apply(ndArray2);
        Assert.assertEquals(result.getFloat(), arithmeticOp.apply(x1, x2).floatValue());
        if (inplace) {
            Assert.assertSame(scalarNDArray, result);
        }
    }

    private void testScalarCornerCase(
            NDManager manager,
            BiFunction<NDArray, Number, NDArray> arrayArithmeticOp,
            BiFunction<Float, Float, Float> arithmeticOp,
            boolean inplace) {
        // scalar with number
        float x1 = 20f;
        float x2 = 4f;
        NDArray ndArray = manager.create(x1);
        NDArray result = arrayArithmeticOp.apply(ndArray, x2);
        Assertions.assertAlmostEquals(result.getFloat(), arithmeticOp.apply(x1, x2));
        if (inplace) {
            Assert.assertSame(ndArray, result);
        } else {
            // zero-dim with number
            ndArray = manager.create(new Shape(2, 0));
            NDArray expected = manager.create(new Shape(2, 0));
            Assert.assertEquals(arrayArithmeticOp.apply(ndArray, x2), expected);
        }
    }

    private void testReverseScalarCornerCase(
            NDManager manager,
            BiFunction<Number, NDArray, NDArray> arrayArithmeticOp,
            BiFunction<Float, Float, Float> arithmeticOp,
            boolean inplace) {
        // number with scalar
        float x1 = 9f;
        float x2 = 3f;
        NDArray ndArray = manager.create(x1);
        NDArray result = arrayArithmeticOp.apply(x2, ndArray);
        Assert.assertEquals(result.getFloat(), arithmeticOp.apply(x2, x1).floatValue());
        if (inplace) {
            Assert.assertSame(ndArray, result);
        } else {
            // number with zero-dim
            ndArray = manager.create(new Shape(0, 2, 3));
            NDArray expected = manager.create(new Shape(0, 2, 3));
            Assert.assertEquals(arrayArithmeticOp.apply(x2, ndArray), expected);
        }
    }

    @Test
    public void testAddScalar() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());
            NDManager manager = model.getNDManager();
            NDArray lhs = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray result;
            try (Trainer trainer =
                    model.newTrainer(
                            new DefaultTrainingConfig(Loss.l2Loss())
                                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT))) {
                try (GradientCollector gradCol = trainer.newGradientCollector()) {
                    lhs.setRequiresGradient(true);
                    result = NDArrays.add(lhs, 2);
                    // check add scalar result
                    gradCol.backward(result);

                    Assert.assertNotEquals(
                            result, lhs, "None in-place operation returned in-place result");
                    NDArray expected = manager.create(new float[] {3f, 4f, 5f, 6f});
                    Assert.assertEquals(
                            result, expected, "AddScala: Incorrect value in summed array");

                    // check add backward
                    NDArray expectedGradient = manager.create(new float[] {1f, 1f, 1f, 1f});
                    Assert.assertEquals(
                            lhs.getGradient(),
                            expectedGradient,
                            "AddScala backward: Incorrect gradient after backward");
                }
            }
            // test inplace
            lhs = manager.create(new float[] {1f, 2f, 3f, 4f});
            result = NDArrays.addi(lhs, 2);
            NDArray expected = manager.create(new float[] {3f, 4f, 5f, 6f});
            Assertions.assertInPlaceEquals(result, expected, lhs);
            testScalarCornerCase(manager, NDArrays::add, Float::sum, false);
            testScalarCornerCase(manager, NDArrays::addi, Float::sum, true);
        }
    }

    @Test
    public void testAddNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray addend = manager.create(new float[] {1f, 2f, 3f, 4f});
            NDArray addendum = manager.create(new float[] {2f, 3f, 4f, 5f});
            NDArray result = NDArrays.add(addend, addendum);
            Assert.assertNotEquals(
                    result, addend, "None in-place operation returned in-place result");

            result = NDArrays.addi(addend, addendum);
            NDArray expected = manager.create(new float[] {3f, 5f, 7f, 9f});
            Assertions.assertInPlaceEquals(result, expected, addend);

            NDArray[] toAddAll = {
                manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2)),
                manager.create(new float[] {4, 3, 2, 1}, new Shape(2, 2)),
                manager.create(new float[] {2, 2, 2, 2}, new Shape(2, 2))
            };

            NDArray addAll = NDArrays.add(toAddAll);
            Assert.assertNotEquals(
                    addAll, toAddAll[0], "None in-place operation returned in-place result");

            addAll = NDArrays.addi(toAddAll);
            Assert.assertEquals(addAll, toAddAll[0], "In-place summation failed");

            expected = manager.create(new float[] {7, 7, 7, 7}, new Shape(2, 2));
            Assert.assertEquals(addAll, expected, "Incorrect value in summed array");

            testCornerCase(manager, NDArrays::add, Float::sum, false);
            testCornerCase(manager, NDArrays::addi, Float::sum, true);
        }
    }

    @Test
    public void testSubScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 9, 12, 11, 0});
            NDArray result = NDArrays.sub(minuend, 3);
            NDArray inPlaceResult = NDArrays.subi(minuend, 3);
            NDArray expected = manager.create(new float[] {3, 6, 9, 8, -3});
            Assert.assertEquals(
                    result, expected, "Scalar subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, minuend);

            testScalarCornerCase(manager, NDArrays::sub, (x, y) -> x - y, false);
            testScalarCornerCase(manager, NDArrays::subi, (x, y) -> x - y, true);
        }
    }

    @Test
    public void testSubNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 9, 12, 15, 0});
            NDArray subtrahend = manager.create(new float[] {2, 3, 4, 5, 6});
            NDArray result = NDArrays.sub(minuend, subtrahend);
            NDArray inPlaceResult = NDArrays.subi(minuend, subtrahend);
            NDArray expected = manager.create(new float[] {4, 6, 8, 10, -6});
            Assert.assertEquals(
                    result,
                    expected,
                    "Element wise subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, minuend);

            testCornerCase(manager, NDArrays::sub, (x, y) -> x - y, false);
            testCornerCase(manager, NDArrays::subi, (x, y) -> x - y, true);
        }
    }

    @Test
    public void testReverseSubScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 91, 12, 215, 180});
            NDArray result = NDArrays.sub(180, minuend);
            NDArray inPlaceResult = NDArrays.subi(180, minuend);
            NDArray expected = manager.create(new float[] {174, 89, 168, -35, 0});
            Assert.assertEquals(
                    result,
                    expected,
                    "Scalar reverse subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, minuend);

            testReverseScalarCornerCase(manager, NDArrays::sub, (x, y) -> x - y, false);
            testReverseScalarCornerCase(manager, NDArrays::subi, (x, y) -> x - y, true);
        }
    }

    @Test
    public void testReverseSubNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray minuend = manager.create(new float[] {6, 9, 12, 15, 45});
            NDArray subtrahend = manager.create(new float[] {24, 63, 96, 15, 90});
            NDArray result = minuend.getNDArrayInternal().rsub(subtrahend);
            NDArray inPlaceResult = minuend.getNDArrayInternal().rsubi(subtrahend);
            NDArray expected = manager.create(new float[] {18, 54, 84, 0, 45});
            Assert.assertEquals(
                    result,
                    expected,
                    "Reverse Element wise subtraction: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, minuend);

            NDArray scalarNDArray = manager.create(5f);
            testReverseCornerCase(
                    manager,
                    scalarNDArray,
                    scalarNDArray.getNDArrayInternal()::rsub,
                    (x, y) -> y - x,
                    false);
            testReverseCornerCase(
                    manager,
                    scalarNDArray,
                    scalarNDArray.getNDArrayInternal()::rsubi,
                    (x, y) -> y - x,
                    true);

            NDArray ndArray1 = manager.create(new Shape(4, 0, 1));
            NDArray ndArray2 = manager.create(new Shape(1, 0));
            Assert.assertEquals(
                    ndArray1.getNDArrayInternal().rsub(ndArray2),
                    manager.create(new Shape(4, 0, 0)));

            ndArray1 = manager.create(new float[] {10f});
            ndArray2 = manager.create(new Shape(2, 0, 3));
            Assert.assertEquals(
                    ndArray1.getNDArrayInternal().rsub(ndArray2),
                    manager.create(new Shape(2, 0, 3)));
        }
    }

    @Test
    public void testMulScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray multiplicand = manager.create(new float[] {6, 9, -12, 15, 0});
            NDArray result = NDArrays.mul(multiplicand, 3);
            NDArray inPlaceResult = NDArrays.muli(multiplicand, 3);
            NDArray expected = manager.create(new float[] {18, 27, -36, 45, 0});
            Assert.assertEquals(
                    result, expected, "Scalar multiplication: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, multiplicand);
            testScalarCornerCase(manager, NDArrays::mul, (x, y) -> x * y, false);
            testScalarCornerCase(manager, NDArrays::muli, (x, y) -> x * y, true);
        }
    }

    @Test
    public void testMulNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray multiplicand = manager.create(new float[] {6, 9, 12, 15, 0});
            NDArray with = manager.create(new float[] {2, 3, 4, 5, 6});
            NDArray result = NDArrays.mul(multiplicand, with);
            NDArray inPlaceResult = NDArrays.muli(multiplicand, with);
            NDArray expected = manager.create(new float[] {12, 27, 48, 75, 0});
            Assert.assertEquals(
                    result,
                    expected,
                    "Element wise multiplication: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, multiplicand);

            NDArray[] toMulAll = {
                manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2)),
                manager.create(new float[] {4, 3, 2, 1}, new Shape(2, 2)),
                manager.create(new float[] {2, 2, 2, 2}, new Shape(2, 2))
            };
            NDArray mulAll = NDArrays.mul(toMulAll);
            NDArray mulAllInPlace = NDArrays.muli(toMulAll);
            Assert.assertNotSame(
                    mulAll, toMulAll[0], "None in-place operation returned in-place result");
            Assert.assertEquals(mulAllInPlace, toMulAll[0], "In-place summation failed");
            expected = manager.create(new float[] {8, 12, 12, 8}, new Shape(2, 2));
            Assert.assertEquals(mulAll, expected, "Incorrect value in summed array");
            Assert.assertEquals(mulAllInPlace, expected, "Incorrect value in summed array");

            testCornerCase(manager, NDArrays::mul, (x, y) -> x * y, false);
            testCornerCase(manager, NDArrays::muli, (x, y) -> x * y, true);
        }
    }

    @Test
    public void testDot() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());
            NDManager manager = model.getNDManager();
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            NDArray result;
            NDArray expected;
            // test 2D * 2D
            try (Trainer trainer =
                    model.newTrainer(
                            new DefaultTrainingConfig(Loss.l2Loss())
                                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT))) {
                try (GradientCollector gradCol = trainer.newGradientCollector()) {
                    lhs.setRequiresGradient(true);
                    result = NDArrays.dot(lhs, rhs);
                    gradCol.backward(result);
                    expected = manager.create(new float[] {33, 14}, new Shape(2, 1));
                    Assert.assertEquals(
                            result,
                            expected,
                            "Matrix multiplication: Incorrect value in result ndarray");

                    NDArray expectedGradient =
                            manager.create(new float[] {2, 3, -4, 2, 3, -4}, new Shape(2, 3));
                    Assert.assertEquals(
                            lhs.getGradient(),
                            expectedGradient,
                            "Matrix multiplication: Incorrect gradient after backward");
                }
            }
            // test 1D * 1D
            lhs = manager.create(new float[] {1f, 2f});
            rhs = manager.create(new float[] {3f, 4f});
            expected = manager.create(11f);
            Assert.assertEquals(lhs.dot(rhs), expected);
            Assert.assertEquals(NDArrays.dot(lhs, rhs), expected);
            // test scalar * ND
            lhs = manager.create(3f);
            rhs = manager.create(new float[] {2f, 3f});
            expected = manager.create(new float[] {6f, 9f});
            Assert.assertEquals(lhs.dot(rhs), expected);
            Assert.assertEquals(NDArrays.dot(lhs, rhs), expected);
            // test 1D * ND
            lhs = manager.create(new float[] {1f, 2f});
            rhs = manager.arange(1.0f, 5.0f).reshape(2, 2);
            expected = manager.create(new float[] {7f, 10f});
            Assert.assertEquals(lhs.dot(rhs), expected);
            Assert.assertEquals(NDArrays.dot(lhs, rhs), expected);
            // test MD * ND
            lhs = manager.create(new float[] {1f, 2f}, new Shape(2, 1));
            rhs = manager.arange(1.0f, 5.0f).reshape(2, 1, 2);
            expected =
                    manager.create(
                            new float[] {1f, 2f, 3f, 4f, 2f, 4f, 6f, 8f}, new Shape(2, 2, 2));
            Assert.assertEquals(lhs.dot(rhs), expected);
            Assert.assertEquals(NDArrays.dot(lhs, rhs), expected);
            // scalar
            lhs = manager.create(4f);
            rhs = manager.create(2f);
            expected = manager.create(8f);
            Assert.assertEquals(lhs.dot(rhs), expected);
            Assert.assertEquals(NDArrays.dot(lhs, rhs), expected);
            // zero-dim
            lhs = manager.create(new Shape(2, 0));
            rhs = manager.create(new Shape(0, 2));
            expected = manager.zeros(new Shape(2, 2));
            Assert.assertEquals(lhs.dot(rhs), expected);
            Assert.assertEquals(NDArrays.dot(lhs, rhs), expected);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testMatMul() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // 2D * 2D
            NDArray lhs = manager.create(new float[] {5, 10, -3, 4, 2, 7}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 9, 3}, new Shape(3, 1));
            NDArray expected = manager.create(new float[] {91, 47}, new Shape(2, 1));
            Assert.assertEquals(lhs.matMul(rhs), expected);
            // 3D * 2D
            lhs = manager.arange(24f).reshape(2, 3, 4);
            rhs = manager.arange(12f).reshape(4, 3);
            expected =
                    manager.create(
                            new float[] {
                                42, 48, 54, 114, 136, 158, 186, 224, 262, 258, 312, 366, 330, 400,
                                470, 402, 488, 574
                            },
                            new Shape(2, 3, 3));
            Assert.assertEquals(lhs.matMul(rhs), expected);
            // 1D * 2D
            lhs = manager.create(new float[] {2f, 7f});
            rhs = manager.arange(6f).reshape(2, 3);
            expected = manager.create(new float[] {21, 30, 39});
            Assert.assertEquals(lhs.matMul(rhs), expected);
            // scalar case, throw exception
            lhs = manager.create(1f);
            rhs = manager.arange(6f).reshape(2, 3);
            expected = manager.create(new float[] {21, 30, 39});
            Assert.assertEquals(lhs.matMul(rhs), expected);
            // zero-dim
            lhs = manager.create(new Shape(0, 3));
            rhs = manager.create(new Shape(3, 0));
            expected = manager.create(new Shape(0, 0));
            Assert.assertEquals(lhs.matMul(rhs), expected);
            lhs = manager.create(new Shape(3, 0));
            rhs = manager.create(new Shape(0, 2));
            expected = manager.zeros(new Shape(3, 2));
            Assert.assertEquals(lhs.matMul(rhs), expected);
        }
    }

    @Test
    public void testDivScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 0});
            NDArray result = NDArrays.div(dividend, 3);
            NDArray inPlaceResult = NDArrays.divi(dividend, 3);
            NDArray expected = manager.create(new float[] {2, 3, 4, 5, 0});
            Assert.assertEquals(
                    result, expected, "Scalar division: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);

            testScalarCornerCase(manager, NDArrays::div, (x, y) -> x / y, false);
            testScalarCornerCase(manager, NDArrays::divi, (x, y) -> x / y, true);
        }
    }

    @Test
    public void testDivNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 0});
            NDArray divisor = manager.create(new float[] {2, 3, 4, 5, 6});
            NDArray result = NDArrays.div(dividend, divisor);
            NDArray inPlaceResult = NDArrays.divi(dividend, divisor);
            NDArray expected = manager.create(new float[] {3, 3, 3, 3, 0});
            Assert.assertEquals(
                    result, expected, "Element wise Division: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);

            testCornerCase(manager, NDArrays::div, (x, y) -> x / y, false);
            testCornerCase(manager, NDArrays::divi, (x, y) -> x / y, true);
        }
    }

    @Test
    public void testReverseDivScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 45});
            NDArray result = NDArrays.div(180, dividend);
            NDArray inPlaceResult = NDArrays.divi(180, dividend);
            NDArray expected = manager.create(new float[] {30, 20, 15, 12, 4});
            Assert.assertEquals(
                    result, expected, "Scalar reverse division: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);

            testReverseScalarCornerCase(manager, NDArrays::sub, (x, y) -> x - y, false);
            testReverseScalarCornerCase(manager, NDArrays::subi, (x, y) -> x - y, true);
        }
    }

    @Test
    public void testReverseDivNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {6, 9, 12, 15, 45});
            NDArray divisor = manager.create(new float[] {24, 63, 96, 15, 90});
            NDArray result = dividend.getNDArrayInternal().rdiv(divisor);
            NDArray inPlaceResult = dividend.getNDArrayInternal().rdivi(divisor);
            NDArray expected = manager.create(new float[] {4, 7, 8, 1, 2});
            Assert.assertEquals(
                    result,
                    expected,
                    "Reverse Element wise Division: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);

            NDArray scalarNDArray = manager.create(24f);
            testReverseCornerCase(
                    manager,
                    scalarNDArray,
                    scalarNDArray.getNDArrayInternal()::rdiv,
                    (x, y) -> y / x,
                    false);
            testReverseCornerCase(
                    manager,
                    scalarNDArray,
                    scalarNDArray.getNDArrayInternal()::rdivi,
                    (x, y) -> y / x,
                    true);
        }
    }

    @Test
    public void testModScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {5, 6, 7, 8, 9});
            NDArray result = NDArrays.mod(dividend, 3);
            NDArray inPlaceResult = NDArrays.modi(dividend, 3);
            NDArray expected = manager.create(new float[] {2, 0, 1, 2, 0});
            Assert.assertEquals(
                    result, expected, "Scalar Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);

            testScalarCornerCase(manager, NDArray::mod, (x, y) -> x % y, false);
            testScalarCornerCase(manager, NDArray::modi, (x, y) -> x % y, true);
        }
    }

    @Test
    public void testModNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {7, 8, 9, 10, 11});
            NDArray divisor = manager.create(new float[] {2, 3, 4, 5, 6});
            NDArray result = NDArrays.mod(dividend, divisor);
            NDArray inPlaceResult = NDArrays.modi(dividend, divisor);
            NDArray expected = manager.create(new float[] {1, 2, 1, 0, 5});
            Assert.assertEquals(
                    result, expected, "Element wise Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);

            testCornerCase(manager, NDArrays::mod, (x, y) -> x % y, false);
            testCornerCase(manager, NDArrays::modi, (x, y) -> x % y, true);
        }
    }

    @Test
    public void testReverseModScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {5, 6, 7, 8, 9});
            NDArray result = NDArrays.mod(180, dividend);
            NDArray inPlaceResult = NDArrays.modi(180, dividend);
            NDArray expected = manager.create(new float[] {0, 0, 5, 4, 0});
            Assert.assertEquals(
                    result,
                    expected,
                    "Scalar reverse Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);

            testReverseScalarCornerCase(manager, NDArrays::mod, (x, y) -> x % y, false);
            testReverseScalarCornerCase(manager, NDArrays::modi, (x, y) -> x % y, true);
        }
    }

    @Test
    public void testReverseModNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray dividend = manager.create(new float[] {7, 8, 9, 10, 11});
            NDArray divisor = manager.create(new float[] {20, 21, 22, 23, 24});
            NDArray result = dividend.getNDArrayInternal().rmod(divisor);
            NDArray inPlaceResult = dividend.getNDArrayInternal().rmodi(divisor);
            NDArray expected = manager.create(new float[] {6, 5, 4, 3, 2});
            Assert.assertEquals(
                    result,
                    expected,
                    "Reverse Element wise Remainder: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, dividend);
            NDArray scalarNDArray = manager.create(20f);
            testReverseCornerCase(
                    manager,
                    scalarNDArray,
                    scalarNDArray.getNDArrayInternal()::rmod,
                    (x, y) -> y % x,
                    false);
            testReverseCornerCase(
                    manager,
                    scalarNDArray,
                    scalarNDArray.getNDArrayInternal()::rmodi,
                    (x, y) -> y % x,
                    true);
        }
    }

    @Test
    public void testPowScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {6, 0, -1, 5, 2}, new Shape(1, 5));
            NDArray result = array.pow(2);
            NDArray inPlaceResult = array.powi(2);
            NDArray expected = manager.create(new float[] {36, 0, 1, 25, 4}, new Shape(1, 5));
            Assertions.assertAlmostEquals(result, expected);
            Assertions.assertInPlaceAlmostEquals(inPlaceResult, expected, array);

            testScalarCornerCase(manager, NDArray::pow, (x, y) -> (float) Math.pow(x, y), false);
            testScalarCornerCase(manager, NDArray::powi, (x, y) -> (float) Math.pow(x, y), true);
        }
    }

    @Test
    public void testPowNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {6, 9, 12, 2, 0});
            NDArray power = manager.create(new float[] {3, 0, 1, -2, 3});
            NDArray result = array.pow(power);
            NDArray inPlaceResult = array.powi(power);
            NDArray expected = manager.create(new float[] {216, 1, 12, 0.25f, 0});
            Assert.assertEquals(
                    result, expected, "Scalar power: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, array);

            testCornerCase(manager, NDArrays::pow, (x, y) -> (float) Math.pow(x, y), false);
            testCornerCase(manager, NDArrays::powi, (x, y) -> (float) Math.pow(x, y), true);
        }
    }

    @Test
    public void testReversePowScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new float[] {3, 4, 5, 6, 7});
            NDArray power = NDArrays.pow(2, array);
            NDArray inPlaceResult = NDArrays.powi(2, array);
            NDArray expected = manager.create(new float[] {8, 16, 32, 64, 128});
            Assert.assertEquals(
                    power, expected, "Scalar reverse power: Incorrect value in result ndarray");
            Assertions.assertInPlaceEquals(inPlaceResult, expected, array);

            testReverseScalarCornerCase(
                    manager, NDArrays::pow, (x, y) -> (float) Math.pow(x, y), false);
            testReverseScalarCornerCase(
                    manager, NDArrays::powi, (x, y) -> (float) Math.pow(x, y), true);
        }
    }

    @Test
    public void testBatchDot() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.ones(new Shape(2, 1, 4));
            NDArray array2 = manager.ones(new Shape(2, 4, 6));
            NDArray expected = manager.create(4f).tile(12).reshape(new Shape(2, 1, 6));
            Assert.assertEquals(
                    array1.batchDot(array2),
                    expected,
                    "batch dot product: Incorrect value in result ndarray");
        }
    }
}
