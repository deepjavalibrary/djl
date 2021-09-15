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
package ai.djl.mxnet.engine;

/** An enum that indicates whether gradient is required. */
public enum GradReq {
    NULL("null", 0),
    WRITE("write", 1),
    ADD("add", 3);

    private String type;
    private int value;

    GradReq(String type, int value) {
        this.type = type;
        this.value = value;
    }

    /**
     * Gets the type of this {@code GradReq}.
     *
     * @return the type
     */
    public String getType() {
        return type;
    }

    /**
     * Gets the value of this {@code GradType}.
     *
     * @return the value
     */
    public int getValue() {
        return value;
    }
}
