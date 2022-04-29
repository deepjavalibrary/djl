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

import com.google.gson.annotations.SerializedName;

/** A class represents a metric dimension. */
public class Dimension {

    @SerializedName("Name")
    private String name;

    @SerializedName("Value")
    private String value;

    /** Constructs a new {@code Dimension} instance. */
    public Dimension() {}

    /**
     * Constructs a new {@code Dimension} instance.
     *
     * @param name the dimension name
     * @param value the dimension value
     */
    public Dimension(String name, String value) {
        this.name = name;
        this.value = value;
    }

    /**
     * Returns the dimension name.
     *
     * @return the dimension name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the dimension value.
     *
     * @return the dimension value
     */
    public String getValue() {
        return value;
    }
}
