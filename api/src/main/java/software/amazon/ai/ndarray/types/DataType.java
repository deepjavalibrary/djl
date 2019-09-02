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
    FLOAT32(Format.FLOATING, 4),
    FLOAT64(Format.FLOATING, 8),
    FLOAT16(Format.FLOATING, 2),
    UINT8(Format.UINT, 1),
    INT32(Format.INT, 4),
    INT8(Format.INT, 1),
    INT64(Format.INT, 8),
    UNKNOWN(Format.UNKNOWN, 0);

    public enum Format {
        FLOATING,
        UINT,
        INT,
        UNKNOWN
    }

    private Format format;
    private int numOfBytes;

    DataType(Format format, int numOfBytes) {
        this.format = format;
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
     * Returns the format of the data type.
     *
     * @return Returns the format of the data type
     */
    public Format getFormat() {
        return format;
    }

    /**
     * Checks whether it is a floating data type.
     *
     * @return {@code true} if it is a floating data type
     */
    public boolean isFloating() {
        return format == Format.FLOATING;
    }

    /**
     * Checks whether it is an integer data type.
     *
     * @return {@code true} if it is an integer type
     */
    public boolean isInteger() {
        return format == Format.UINT || format == Format.INT;
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

    public Buffer asDataType(ByteBuffer data) {
        switch (this) {
            case FLOAT32:
                return data.asFloatBuffer();
            case FLOAT64:
                return data.asDoubleBuffer();
            case INT32:
                return data.asIntBuffer();
            case INT64:
                return data.asLongBuffer();
            case UINT8:
            case INT8:
            case FLOAT16:
            case UNKNOWN:
            default:
                return data;
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
