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

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

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
     * @return {@code true} if it is a real type
     */
    public boolean isReal() {
        return type.startsWith("float");
    }
    /**
     * Checks whether it is an integer data type.
     *
     * @return {@code true} if it is an integer type
     */
    public boolean isInteger() {
        return type.startsWith("int") || type.startsWith("uint");
    }

    public static DataType fromBuffer(Buffer data) {
        if (data instanceof FloatBuffer) {
            return DataType.FLOAT32;
        } else if (data instanceof DoubleBuffer) {
            return DataType.FLOAT64;
        } else if (data instanceof IntBuffer) {
            return DataType.INT32;
        } else if (data instanceof LongBuffer) {
            return DataType.INT64;
        } else if (data instanceof ByteBuffer) {
            return DataType.INT8;
        } else {
            throw new IllegalArgumentException(
                    "Unsupported buffer type: " + data.getClass().getSimpleName());
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return type;
    }
}
