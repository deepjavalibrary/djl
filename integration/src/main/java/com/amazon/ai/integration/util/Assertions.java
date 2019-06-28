package com.amazon.ai.integration.util;

import org.apache.mxnet.engine.MxNDArray;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;

public final class Assertions {

    private Assertions() {}

    public static void assertEquals(boolean statement, String errorMessage)
            throws FailedTestException {
        if (!statement) {
            throw new FailedTestException(errorMessage);
        }
    }

    public static void assertEquals(boolean statement) throws FailedTestException {
        assertEquals(statement, "Assertion failed!");
    }

    public static void assertEquals(NDArray expected, NDArray actual, String errorMessage)
            throws FailedTestException {
        if (!NDArrays.equals(expected, actual)) {
            throw new FailedTestException(errorMessage);
        }
    }

    public static void assertEquals(NDArray expected, NDArray actual) throws FailedTestException {
        assertEquals(expected, actual, "Two NDArrays are different!");
    }

    public static void assertNonZeroNumber(NDArray array, int number, String errorMessage)
            throws FailedTestException {
        if (array.nonzero() != number) {
            throw new FailedTestException(errorMessage);
        }
    }

    public static void assertNonZeroNumber(NDArray array, int number) throws FailedTestException {
        assertNonZeroNumber(array, number, "Assertion failed!");
    }

    public static void assertInPlace(MxNDArray expected, MxNDArray actual, String errorMessage)
            throws FailedTestException {
        if (!expected.getHandle().equals(actual.getHandle())) {
            throw new FailedTestException(errorMessage);
        }
    }

    public static void assertInPlace(MxNDArray expected, MxNDArray actual)
            throws FailedTestException {
        assertInPlace(expected, actual, "Assertion failed!");
    }
}
