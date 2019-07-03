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

/** An enum representing the underlying {@link software.amazon.ai.ndarray.NDArray}'s data type. */
public enum DataType {
    FLOAT32("float32", 4),
    FLOAT64("float64", 8),
    FLOAT16("float16", 2),
    UINT8("uint8", 1),
    INT32("int32", 4),
    INT8("int8", 1),
    INT64("int64", 8),
    UNKNOWN("unknown", 0);

    private String type;
    private int numOfBytes;

    DataType(String type, int numOfBytes) {
        this.type = type;
        this.numOfBytes = numOfBytes;
    }

    /**
     * Returns number of bytes for each element.
     *
     * @return number of bytes for each element
     */
    public int getNumOfBytes() {
        return numOfBytes;
    }

    /**
     * Returns name of the data type.
     *
     * @return name of the data type
     */
    public String getType() {
        return type;
    }

    /**
     * Checks whether it is a real data type.
     *
     * @return true if it is a real type
     */
    public boolean isReal() {
        return type.startsWith("float");
    }
    /**
     * Checks whether it is an integer data type.
     *
     * @return true if it is an integer type
     */
    public boolean isInteger() {
        return type.startsWith("int") || type.startsWith("uint");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return type;
    }
}
