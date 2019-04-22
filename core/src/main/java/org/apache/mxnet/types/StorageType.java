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
package org.apache.mxnet.types;

public enum StorageType {
    UNDEFINED("undefined", -1),
    DEFAULT("default", 0),
    ROW_SPARSE("row_sparse", 1),
    CSR("csr", 2);

    private String type;
    private int value;

    StorageType(String type, int value) {
        this.type = type;
        this.value = value;
    }

    public static StorageType fromValue(int value) {
        for (StorageType t : values()) {
            if (value == t.getValue()) {
                return t;
            }
        }
        throw new IllegalArgumentException("Unknown storage type: " + value);
    }

    public String getType() {
        return type;
    }

    public int getValue() {
        return value;
    }
}
