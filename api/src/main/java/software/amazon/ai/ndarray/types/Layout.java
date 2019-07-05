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

/** A enum representing the underlying {@link software.amazon.ai.ndarray.NDArray}'s layout. */
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

    /**
     * Returns index of 'N' (batch) axis of the {@code Layout}.
     *
     * @param layout {@code Layout} to exam
     * @return index of 'N' (batch) axis of the {@code Layout}
     */
    public static int getBatchAxis(Layout layout) {
        if (layout == null || Layout.UNDEFINED == layout) {
            return 0;
        }

        if (!layout.getValue().contains("N")) {
            throw new IllegalArgumentException("no Batch Axis('N') found in Layout!");
        }

        return layout.getValue().indexOf('N');
    }
}
