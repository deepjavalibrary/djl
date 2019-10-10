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
package ai.djl.mxnet.dataset;

import ai.djl.ndarray.NDArray;

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

    public static void assertAlmostEquals(NDArray actual, NDArray expected) {
        assertAlmostEquals(actual, expected, RTOL, ATOL);
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
}
