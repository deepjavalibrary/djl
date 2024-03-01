/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.concurrent.ConcurrentHashMap;

/** An interface holds metric unit constants. */
public enum Unit {
    MICROSECONDS("Microseconds"),
    MILLISECONDS("Milliseconds"),
    BYTES("Bytes"),
    KILOBYTES("Kilobytes"),
    MEGABYTES("Megabytes"),
    GIGABYTES("Gigabytes"),
    TERABYTES("Terabytes"),
    BITS("Bits"),
    KILOBITS("Kilobits"),
    MEGABITS("Megabits"),
    GIGABITS("Gigabits"),
    TERABITS("Terabits"),
    PERCENT("Percent"),
    COUNT("Count"),
    BYTES_PER_SECOND("Bytes/Second"),
    KILOBYTES_PER_SECOND("Kilobytes/Second"),
    MEGABYTES_PER_SECOND("Megabytes/Second"),
    GIGABYTES_PER_SECOND("Gigabytes/Second"),
    TERABYTES_PER_SECOND("Terabytes/Second"),
    BITS_PER_SECOND("Bits/Second"),
    KILOBITS_PER_SECOND("Kilobits/Second"),
    MEGABITS_PER_SECOND("Megabits/Second"),
    GIGABITS_PER_SECOND("Gigabits/Second"),
    TERABITS_PER_SECOND("Terabits/Second"),
    COUNT_PER_SECOND("Count/Second"),
    COUNT_PER_ITEM("Count/Item"),
    NONE("None");

    private static final ConcurrentHashMap<String, Unit> MAP = new ConcurrentHashMap<>();

    static {
        for (Unit unit : values()) {
            MAP.put(unit.value, unit);
        }
    }

    private final String value;

    Unit(String value) {
        this.value = value;
    }

    /**
     * Returns the string value of the {@code Unit}.
     *
     * @return the string value of the {@code Unit}
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
    public static Unit fromValue(String value) {
        Unit ret = MAP.get(value);
        if (ret == null) {
            throw new IllegalArgumentException("Invalid unit value: " + value);
        }
        return ret;
    }
}
