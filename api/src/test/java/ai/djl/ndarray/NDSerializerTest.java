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
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Float16Utils;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDSerializerTest {

    @Test
    public void testNDSerializer() throws IOException {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            byte[] data = readFile("0d.npy");
            NDArray array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.FLOAT64);
            Assert.assertEquals(array.getShape().dimension(), 0);
            Assert.assertEquals(array.toDoubleArray(), new double[] {1d});
            byte[] buf = encode(array);
            Assert.assertEquals(buf, data);

            data = readFile("1d.npy");
            array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.INT64);
            Assert.assertEquals(array.getShape().dimension(), 1);
            Assert.assertEquals(array.toLongArray(), new long[] {1L});
            buf = encode(array);
            Assert.assertEquals(buf, data);

            data = readFile("2d.npy");
            array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.INT32);
            Assert.assertEquals(array.getShape().dimension(), 2);
            Assert.assertEquals(array.toIntArray(), new int[] {1, 1});
            buf = encode(array);
            Assert.assertEquals(buf, data);

            data = readFile("boolean.npy");
            array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.BOOLEAN);
            Assert.assertEquals(array.getShape().size(), 1);
            Assert.assertEquals(array.toBooleanArray(), new boolean[] {true});
            buf = encode(array);
            Assert.assertEquals(buf, data);

            data = readFile("fp16.npy");
            array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.FLOAT16);
            Assert.assertEquals(array.getShape().size(), 1);
            float[] values = Float16Utils.fromByteBuffer(array.toByteBuffer());
            Assert.assertEquals(values[0], 1f);
            buf = encode(array);
            Assert.assertEquals(buf, data);

            data = readFile("fp32.npy");
            array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.FLOAT32);
            Assert.assertEquals(array.getShape().size(), 1);
            Assert.assertEquals(array.toFloatArray(), new float[] {1f});
            buf = encode(array);
            Assert.assertEquals(buf, data);

            data = readFile("int8.npy");
            array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.INT8);
            Assert.assertEquals(array.getShape().size(), 1);
            Assert.assertEquals(array.toByteArray(), new byte[] {1});
            buf = encode(array);
            Assert.assertEquals(buf, data);

            data = readFile("uint8.npy");
            array = decode(manager, data);
            Assert.assertEquals(array.getDataType(), DataType.UINT8);
            Assert.assertEquals(array.getShape().size(), 1);
            Assert.assertEquals(array.toByteArray(), new byte[] {1});
            buf = encode(array);
            Assert.assertEquals(buf, data);
        }
    }

    private static byte[] encode(NDArray array) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            NDSerializer.encodeAsNumpy(array, bos);
            bos.flush();
            return bos.toByteArray();
        }
    }

    private static NDArray decode(NDManager manager, byte[] data) throws IOException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(data)) {
            return NDSerializer.decodeNumpy(manager, bis);
        }
    }

    static byte[] readFile(String fileName) throws IOException {
        return Files.readAllBytes(Paths.get("src/test/resources/" + fileName));
    }
}
