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
package software.amazon.ai.integration.util;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;

public final class Assertions {

    private Assertions() {}

    private static <T> String getDefaultErrorMessage(T actual, T expected) {
        return getDefaultErrorMessage(actual, expected, null);
    }

    private static <T> String getDefaultErrorMessage(T actual, T expected, String errorMessage) {
        StringBuilder sb = new StringBuilder(100);
        if (errorMessage != null) {
            sb.append(errorMessage);
        }
        sb.append(System.lineSeparator())
                .append("Expected: ")
                .append(expected)
                .append(System.lineSeparator())
                .append("Actual: ")
                .append(actual);
        return sb.toString();
    }

    public static void assertTrue(boolean statement, String errorMessage) {
        if (!statement) {
            throw new AssertionError(errorMessage);
        }
    }

    public static void assertTrue(boolean statement) {
        assertTrue(statement, "Statement is not True!");
    }

    public static void assertFalse(boolean statement, String errorMessage) {
        if (statement) {
            throw new AssertionError(errorMessage);
        }
    }

    public static void assertFalse(boolean statement) {
        assertFalse(statement, "Statement is not False!");
    }

    public static void assertEquals(NDList actual, NDList expected, String errorMessage) {
        assertEquals(
                actual.size(),
                expected.size(),
                getDefaultErrorMessage(actual.size(), expected.size(), errorMessage));
        int size = expected.size();
        for (int i = 0; i < size; i++) {
            assertEquals(actual.size(), expected.size(), "The NDLists differ on element " + i);
        }
    }

    public static void assertEquals(NDList actual, NDList expected) {
        assertEquals(actual, expected, "Two NDArrays are different!");
    }

    public static void assertEquals(NDArray actual, NDArray expected, String errorMessage) {
        if (!NDArrays.equals(actual, expected)) {
            throw new AssertionError(getDefaultErrorMessage(actual, expected, errorMessage));
        }
    }

    public static void assertEquals(NDArray[] actual, NDArray[] expected, String errorMessage) {
        assertEquals(
                actual.length,
                expected.length,
                getDefaultErrorMessage(actual.length, expected.length, errorMessage));
        int size = expected.length;
        for (int i = 0; i < size; i++) {
            assertEquals(actual[i], expected[i], "The NDArrays differ on element " + i);
        }
    }

    public static void assertEquals(NDArray actual, NDArray expected) {
        assertEquals(actual, expected, "Two NDArrays are different!");
    }

    public static <T extends Number> void assertEquals(T actual, T expected) {
        assertEquals(actual, expected, "Two numbers are different!");
    }

    public static <T extends Number> void assertEquals(T actual, T expected, String errorMessage) {
        if (!actual.equals(expected)) {
            throw new AssertionError(getDefaultErrorMessage(actual, expected, errorMessage));
        }
    }

    public static void assertEquals(Parameter actual, Parameter expected) {
        if (!actual.equals(expected)) {
            throw new AssertionError(
                    getDefaultErrorMessage(actual, expected, "Two Parameters are different!"));
        }
    }

    public static void assertEquals(Shape actual, Shape expected) {
        if (!actual.equals(expected)) {
            throw new AssertionError(
                    getDefaultErrorMessage(actual, expected, "Two Shapes are different!"));
        }
    }

    public static void assertAlmostEquals(NDArray actual, NDArray expected) {
        assertAlmostEquals(actual, expected, 1e-5, 1e-3);
    }

    public static void assertAlmostEquals(NDList actual, NDList expected) {
        assertAlmostEquals(actual, expected, 1e-5, 1e-3);
    }

    public static void assertAlmostEquals(
            NDList actual, NDList expected, double rtol, double atol) {
        assertEquals(
                actual.size(),
                expected.size(),
                getDefaultErrorMessage(
                        actual.size(), expected.size(), "The NDLists have different sizes"));
        int size = actual.size();
        for (int i = 0; i < size; i++) {
            assertAlmostEquals(actual.get(i), expected.get(i), rtol, atol);
        }
    }

    public static void assertAlmostEquals(
            NDArray actual, NDArray expected, double rtol, double atol) {
        if (!actual.getShape().equals(expected.getShape())) {
            throw new AssertionError(
                    getDefaultErrorMessage(
                            actual.getShape(),
                            expected.getShape(),
                            "The shape of two NDArray are different!"));
        }
        Number[] actualDoubleArray = actual.toArray();
        Number[] expectedDoubleArray = expected.toArray();
        for (int i = 0; i < actualDoubleArray.length; i++) {
            double a = actualDoubleArray[i].doubleValue();
            double b = expectedDoubleArray[i].doubleValue();
            if (Math.abs(a - b) > (atol + rtol * Math.abs(b))) {
                throw new AssertionError(getDefaultErrorMessage(actual, expected));
            }
        }
    }

    public static void assertInPlace(NDArray actual, NDArray expected, String errorMessage) {
        if (actual != expected) {
            throw new AssertionError(errorMessage);
        }
    }

    public static void assertInPlace(NDArray actual, NDArray expected) {
        assertInPlace(
                actual, expected, getDefaultErrorMessage(actual, expected, "Assertion failed!"));
    }

    public static void assertInPlaceEquals(NDArray actual, NDArray expected, NDArray original) {
        assertEquals(
                actual, expected, getDefaultErrorMessage(actual, expected, "Assert Equal failed!"));
        assertInPlace(
                original,
                expected,
                getDefaultErrorMessage(original, expected, "Assert Inplace failed!"));
    }

    public static void assertInPlaceAlmostEquals(
            NDArray actual, NDArray expected, NDArray original) {
        assertInPlaceAlmostEquals(actual, expected, original, 1e-5, 1e-3);
    }

    public static void assertInPlaceAlmostEquals(
            NDArray actual, NDArray expected, NDArray original, double rtol, double atol) {
        assertAlmostEquals(actual, expected, rtol, atol);
        assertInPlace(
                original,
                expected,
                getDefaultErrorMessage(original, expected, "Assert Inplace failed!"));
    }

    @SuppressWarnings({"PMD.PreserveStackTrace", "PMD.DoNotUseThreads"})
    public static void assertThrows(
            Runnable function, Class<?> exceptionClass, String errorMessage) {
        try {
            function.run();
        } catch (Exception e) {
            if (exceptionClass.isInstance(e)) {
                return;
            } else {
                throw new AssertionError(errorMessage + " - wrong exception type thrown");
            }
        }
        throw new AssertionError(errorMessage + " - did not throw an exception");
    }

    @SuppressWarnings("PMD.DoNotUseThreads")
    public static void assertThrows(Runnable function, Class<?> exceptionClass) {
        assertThrows(function, exceptionClass, "Assertion failed!");
    }
}
