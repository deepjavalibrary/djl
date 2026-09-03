/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray;

import ai.djl.ndarray.types.DataType;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;

public class BaseNDManagerTest {

    /**
     * {@link BaseNDManager#validateBuffer} previously computed the expected byte count as {@code
     * getNumOfBytes() * expected} in 32-bit {@code int}. For large element counts that product
     * overflows and wraps to a small value, so the size check compared against the wrong number.
     * The byte count is now computed in 64-bit, so these element counts are rejected.
     */
    @Test
    public void testValidateBufferIntegerOverflow() {
        ByteBuffer tiny = ByteBuffer.allocate(16);

        // FLOAT32 (4 bytes) * 2^30 == 2^32, which wraps int32 to exactly 0.
        assertOverflowRejected(tiny, DataType.FLOAT32, 1 << 30);

        // FLOAT32 (4 bytes) * (2^30 + 4) == 4294967312, which wraps int32 to a small positive value
        // rather than to zero.
        assertOverflowRejected(tiny, DataType.FLOAT32, (1 << 30) + 4);

        // FLOAT64 (8 bytes) * 2^29 == 2^32, which wraps int32 to exactly 0.
        assertOverflowRejected(tiny, DataType.FLOAT64, 1 << 29);
    }

    /** A genuinely undersized buffer (no overflow) must still be rejected. */
    @Test
    public void testValidateBufferRejectsUndersized() {
        ByteBuffer tiny = ByteBuffer.allocate(16);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> BaseNDManager.validateBuffer(tiny.duplicate(), DataType.FLOAT32, 1000));
    }

    /** A correctly sized buffer must pass validation unchanged. */
    @Test
    public void testValidateBufferAcceptsExactSize() {
        // 4 FLOAT32 elements == 16 bytes.
        ByteBuffer exact = ByteBuffer.allocate(16);
        BaseNDManager.validateBuffer(exact, DataType.FLOAT32, 4);
        Assert.assertEquals(exact.remaining(), 16);
    }

    private static void assertOverflowRejected(ByteBuffer buffer, DataType dataType, int expected) {
        Assert.assertThrows(
                "A "
                        + buffer.remaining()
                        + "-byte buffer must not pass validation for "
                        + expected
                        + " "
                        + dataType
                        + " elements",
                IllegalArgumentException.class,
                () -> BaseNDManager.validateBuffer(buffer.duplicate(), dataType, expected));
    }
}
