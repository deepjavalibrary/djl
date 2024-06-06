/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.metric;

/** An enum holds metric type constants. */
public enum MetricType {
    COUNTER("c"),
    GAUGE("g"),
    HISTOGRAM("h");

    private final String value;

    MetricType(String value) {
        this.value = value;
    }

    /**
     * Returns the string value of the {@code MetricType}.
     *
     * @return the string value of the {@code MetricType}
     */
    public String getValue() {
        return value;
    }

    /**
     * Returns {@code Unit} instance from an string value.
     *
     * @param value the String value of Unit
     * @return the {@code Unit}
     */
    public static MetricType of(String value) {
        if ("c".equals(value)) {
            return COUNTER;
        } else if ("g".equals(value)) {
            return GAUGE;
        } else if ("h".equals(value)) {
            return HISTOGRAM;
        }
        throw new IllegalArgumentException("Invalid MetricType value: " + value);
    }
}
