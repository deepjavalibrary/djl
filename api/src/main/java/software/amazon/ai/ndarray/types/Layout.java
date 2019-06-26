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
package software.amazon.ai.ndarray.types;

public enum Layout {
    UNDEFINED("__undefined__"),
    NCHW("NCHW"),
    NTC("NTC"),
    NT("NT"),
    N("N");

    private String value;

    Layout(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public static Layout fromValue(String value) {
        for (Layout layout : values()) {
            if (layout.value.equals(value)) {
                return layout;
            }
        }
        throw new IllegalArgumentException("Invalid layout value: " + value);
    }
}
