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
package com.amazon.ai.util;

public final class Utils {

    public static final boolean DEBUG = true;

    private Utils() {}

    public static <T> int indexOf(T[] arr, T value) {
        if (arr != null) {
            for (int i = 0; i < arr.length; ++i) {
                if (value.equals(arr[i])) {
                    return i;
                }
            }
        }

        return -1;
    }

    public static <T> boolean contains(T[] arr, T value) {
        return indexOf(arr, value) >= 0;
    }

    public static void pad(StringBuilder sb, char c, int count) {
        for (int i = 0; i < count; ++i) {
            sb.append(c);
        }
    }
}
