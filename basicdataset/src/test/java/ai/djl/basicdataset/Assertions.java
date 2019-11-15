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
package ai.djl.basicdataset;

import ai.djl.ndarray.NDArray;

/** A utility class which provides methods to compare two {@link NDArray}s. */
public final class Assertions {

    private static final double RTOL = 1e-5;
    private static final double ATOL = 1e-3;

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

    /**
     * Tests that the actual {@link NDArray} and the expected {@link NDArray} are approximately
     * equal with rtol=1e-5 and atol=1e-3.
     *
     * <p>the formula is absolute(a - b) <= (atol + rtol * absolute(b))
     *
     * @param actual the {@link NDArray} to compare to
     * @param expected the {@link NDArray} to compare
     */
    public static void assertAlmostEquals(NDArray actual, NDArray expected) {
        assertAlmostEquals(actual, expected, RTOL, ATOL);
    }

    /**
     * Tests that the actual {@link NDArray} and the expected {@link NDArray} are approximately
     * equal given rtol and atol.
     *
     * <p>the formula is absolute(a - b) <= (atol + rtol * absolute(b))
     *
     * @param actual the {@link NDArray} to compare to
     * @param expected the {@link NDArray} to compare
     * @param rtol the relative tolerance parameter
     * @param atol The absolute tolerance parameter
     */
    public static void assertAlmostEquals(
            NDArray actual, NDArray expected, double rtol, double atol) {
        if (!actual.allClose(expected, rtol, atol, false)) {
            throw new AssertionError(getDefaultErrorMessage(actual, expected));
        }
    }
}
