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
package ai.djl.ndarray.types;

import java.util.stream.IntStream;

public enum LayoutType {
    BATCH('N'),
    CHANNEL('C'),
    DEPTH('D'),
    HEIGHT('H'),
    WIDTH('W'),
    TIME('T'),
    UNKNOWN('?');

    private char value;

    LayoutType(char value) {
        this.value = value;
    }

    public char getValue() {
        return value;
    }

    public static LayoutType fromValue(char value) {
        for (LayoutType type : LayoutType.values()) {
            if (value == type.value) {
                return type;
            }
        }
        throw new IllegalArgumentException(
                "The value does not match any layoutTypes. Use '?' for Unknown");
    }

    public static LayoutType[] fromValue(String layout) {
        return IntStream.range(0, layout.length())
                .mapToObj(i -> fromValue(layout.charAt(i)))
                .toArray(LayoutType[]::new);
    }

    public static String toString(LayoutType[] layouts) {
        StringBuilder sb = new StringBuilder(layouts.length);
        for (LayoutType layout : layouts) {
            sb.append(layout.getValue());
        }
        return sb.toString();
    }
}
