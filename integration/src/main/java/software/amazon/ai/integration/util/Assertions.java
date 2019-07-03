package software.amazon.ai.integration.util;

import org.apache.mxnet.engine.MxNDArray;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;

public final class Assertions {

    private Assertions() {}

    public static void assertStatement(boolean statement, String errorMessage)
            throws FailedTestException {
        if (!statement) {
            throw new FailedTestException(errorMessage);
        }
    }

    public static void assertStatement(boolean statement) throws FailedTestException {
        assertStatement(statement, "Assertion failed!");
    }

    public static void assertEquals(NDArray expected, NDArray actual, String errorMessage)
            throws FailedTestException {
        if (!NDArrays.equals(expected, actual)) {
            throw new FailedTestException(errorMessage);
        }
    }

    public static void assertEquals(float expected, float actual, String errorMessage)
            throws FailedTestException {
        if (expected != actual) {
            throw new FailedTestException(errorMessage);
        }
    }

    public static void assertEquals(NDArray expected, NDArray actual) throws FailedTestException {
        assertEquals(expected, actual, "Two NDArrays are different!");
    }

    public static void assertAlmostEquals(
            NDArray expected, NDArray actual, double rtol, double atol) throws FailedTestException {
        Number[] expectedDoubleArray = expected.toArray();
        Number[] actualDoubleArray = actual.toArray();
        if (expectedDoubleArray.length != actualDoubleArray.length) {
            throw new FailedTestException("The length of two NDArray are different!");
        }
        for (int i = 0; i < expectedDoubleArray.length; i++) {
            double a = expectedDoubleArray[i].doubleValue();
            double b = actualDoubleArray[i].doubleValue();
            if (Math.abs(a - b) > (atol + rtol * Math.abs(b))) {
                throw new FailedTestException(
                        String.format(
                                "expect = %s, actual = %s", String.valueOf(a), String.valueOf(b)));
            }
        }
    }

    public static void assertAlmostEquals(NDArray expected, NDArray actual)
            throws FailedTestException {
        assertAlmostEquals(expected, actual, 1e-5, 1e-3);
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
