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

import ai.djl.Device;
import ai.djl.ndarray.types.DataType;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;

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

    /**
     * {@link BaseNDManager#validateBuffer} previously compared the element width of the buffer with
     * itself rather than with the data type, so a typed buffer whose elements are narrower or wider
     * than the data type was accepted.
     */
    @Test
    public void testValidateBufferRejectsMismatchedWidth() {
        // INT32 elements (4 bytes) for an INT64 array (8 bytes).
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> BaseNDManager.validateBuffer(IntBuffer.allocate(4), DataType.INT64, 4));
        // FLOAT32 elements (4 bytes) for a FLOAT16 array (2 bytes).
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> BaseNDManager.validateBuffer(FloatBuffer.allocate(4), DataType.FLOAT16, 4));
        // FLOAT16 elements (2 bytes) for a FLOAT32 array (4 bytes).
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> BaseNDManager.validateBuffer(ShortBuffer.allocate(4), DataType.FLOAT32, 4));

        // Buffers whose elements have the width of the data type are still accepted.
        BaseNDManager.validateBuffer(IntBuffer.allocate(4), DataType.UINT32, 4);
        BaseNDManager.validateBuffer(ShortBuffer.allocate(4), DataType.BFLOAT16, 4);
        BaseNDManager.validateBuffer(DoubleBuffer.allocate(4), DataType.FLOAT64, 4);
    }

    /** {@link BaseNDManager#toBufferSize} rejects negative counts and byte counts beyond int. */
    @Test
    public void testToBufferSize() {
        Assert.assertEquals(BaseNDManager.toBufferSize(4, DataType.FLOAT32), 16);
        Assert.assertEquals(BaseNDManager.toBufferSize(0, DataType.FLOAT64), 0);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> BaseNDManager.toBufferSize(-1, DataType.FLOAT32));

        // FLOAT32 (4 bytes) * 2^30 == 2^32, which does not fit in an int.
        Assert.assertThrows(
                ArithmeticException.class,
                () -> BaseNDManager.toBufferSize(1L << 30, DataType.FLOAT32));

        // FLOAT64 (8 bytes) * 2^61 == 2^64, which does not fit in a long.
        Assert.assertThrows(
                ArithmeticException.class,
                () -> BaseNDManager.toBufferSize(1L << 61, DataType.FLOAT64));
    }

    /**
     * The 2D {@code create} overloads previously sized the buffer as {@code rows * cols * bytes} in
     * 32-bit {@code int}, which wraps for large inputs. The byte count is now checked.
     */
    @Test
    public void testCreate2DIntegerOverflow() {
        // 2^16 rows sharing one 2^14 element row: 2^30 elements, 2^32 bytes as FLOAT32.
        float[][] data = new float[1 << 16][];
        Arrays.fill(data, new float[1 << 14]);
        // 2^14 rows sharing one 2^14 element row: 2^28 elements, 2^31 bytes as FLOAT64 or INT64.
        double[][] doubles = new double[1 << 14][];
        Arrays.fill(doubles, new double[1 << 14]);
        long[][] longs = new long[1 << 14][];
        Arrays.fill(longs, new long[1 << 14]);
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            Assert.assertThrows(ArithmeticException.class, () -> manager.create(data));
            Assert.assertThrows(ArithmeticException.class, () -> manager.create(doubles));
            Assert.assertThrows(ArithmeticException.class, () -> manager.create(longs));
        }
    }

    private static void assertOverflowRejected(ByteBuffer buffer, DataType dataType, int expected) {
        Assert.assertThrows(
                ArithmeticException.class,
                () -> BaseNDManager.validateBuffer(buffer.duplicate(), dataType, expected));
    }
}
