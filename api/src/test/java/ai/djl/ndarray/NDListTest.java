/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public class NDListTest {

    @Test
    public void testNumpy() throws IOException {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            byte[] data = NDSerializerTest.readFile("list.npz");
            NDList decoded = NDList.decode(manager, data);

            ByteArrayOutputStream bos = new ByteArrayOutputStream(data.length + 1);
            decoded.encode(bos, NDList.Encoding.NPZ);
            NDList list = NDList.decode(manager, bos.toByteArray());
            Assert.assertEquals(list.size(), 2);
            Assert.assertEquals(list.get(0).getName(), "bool8");
        }
    }

    @Test
    public void testSafetensors() throws IOException {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            byte[] data = NDSerializerTest.readFile("list.safetensors");
            NDList decoded = NDList.decode(manager, data);

            ByteArrayOutputStream bos = new ByteArrayOutputStream(data.length + 1);
            decoded.encode(bos, NDList.Encoding.SAFETENSORS);
            NDList list = NDList.decode(manager, bos.toByteArray());
            Assert.assertEquals(list.size(), 2);
            Assert.assertEquals(list.get(0).getName(), "attention");
            Assert.assertEquals(list.get(0).toByteArray(), new byte[] {0, 1, 2, 3, 4, 5});
        }
    }

    /** Safetensors entries whose data offsets do not match their shape and dtype are rejected. */
    @Test
    public void testSafetensorsMalformedMetadata() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            // 8 FLOAT32 elements need 32 bytes, but the offsets only cover 16.
            byte[] undersized = safetensors("F32", 8, 0, 16);
            Assert.assertThrows(
                    IllegalArgumentException.class, () -> NDList.decode(manager, undersized));

            byte[] negative = safetensors("F32", 4, -4, 12);
            Assert.assertThrows(
                    IllegalArgumentException.class, () -> NDList.decode(manager, negative));

            // FLOAT64 (8 bytes) * 2^29 == 2^32, which does not fit in an int.
            byte[] overflow = safetensors("F64", 1 << 29, 0, 16);
            Assert.assertThrows(ArithmeticException.class, () -> NDList.decode(manager, overflow));
        }
    }

    private static byte[] safetensors(String dtype, int size, int begin, int end) {
        String header =
                "{\"a\":{\"dtype\":\""
                        + dtype
                        + "\",\"shape\":["
                        + size
                        + "],\"data_offsets\":["
                        + begin
                        + ','
                        + end
                        + "]}}";
        byte[] json = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer bb = ByteBuffer.allocate(8 + json.length + end);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(json.length);
        bb.put(json);
        return bb.array();
    }
}
