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

package software.amazon.ai.util;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import software.amazon.ai.ndarray.types.DataType;

public class UtilsTest {

    @BeforeClass
    public void preprocess() {
        Utils.deleteQuietly(Paths.get("build/tmp/testFile/"));
    }

    @AfterClass
    public void postprocess() {
        Utils.deleteQuietly(Paths.get("build/tmp/testFile/"));
    }

    @Test
    public void testContains() {
        String[] words = new String[] {"Hello World", "2"};
        Assert.assertTrue(Utils.contains(words, "2"));
        Assert.assertFalse(Utils.contains(words, "3"));
    }

    @Test
    public void testPad() {
        StringBuilder sb = new StringBuilder();
        sb.append("Hello");
        Utils.pad(sb, 'a', 5);
        Assert.assertEquals(sb.toString(), "Helloaaaaa");
    }

    @Test
    public void testFile() throws IOException {
        Path dir = Paths.get("build/tmp/testFile/");
        Files.createDirectories(dir);
        Path file = dir.resolve("synset.txt");
        Assert.assertTrue(Utils.readLines(file).isEmpty());

        try (BufferedWriter writer = Files.newBufferedWriter(file)) {
            writer.append("line1");
            writer.newLine();
        }
        Assert.assertEquals(Utils.readLines(file).size(), 1);

        Utils.deleteQuietly(dir);
        Assert.assertTrue(Files.notExists(dir));
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testToCharSequence() {
        ByteBuffer buf = ByteBuffer.allocate(32);
        buf.asDoubleBuffer()
                .put(Double.NEGATIVE_INFINITY)
                .put(Double.MAX_VALUE)
                .put(Double.NaN)
                .put(-1d);
        CharSequence str = Utils.toCharSequence(buf, DataType.FLOAT64);
        Assert.assertEquals(
                str.toString(),
                "      -Infinity,  1.7976931e+308,             NaN,  -1.0000000e+00");

        buf.rewind();
        buf.asLongBuffer().put(Long.MAX_VALUE).put(Long.MIN_VALUE).put(1L).put(-1L);
        str = Utils.toCharSequence(buf, DataType.INT64);
        Assert.assertEquals(
                str.toString(),
                "0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0x0000000000000001, 0xFFFFFFFFFFFFFFFF");

        buf.rewind();
        buf.limit(16);
        buf.asFloatBuffer()
                .put(Float.NEGATIVE_INFINITY)
                .put(Float.MAX_VALUE)
                .put(Float.NaN)
                .put(-1f);
        str = Utils.toCharSequence(buf, DataType.FLOAT32);
        Assert.assertEquals(
                str.toString(), "     -Infinity,  3.4028235e+38,            NaN, -1.0000000e+00");

        buf.rewind();
        buf.asIntBuffer().put(Integer.MAX_VALUE).put(Integer.MIN_VALUE).put(1).put(-1);
        str = Utils.toCharSequence(buf, DataType.INT32);
        Assert.assertEquals(str.toString(), " 2147483647, -2147483648,           1,          -1");

        buf.rewind();
        buf.limit(4);
        buf.put(Byte.MAX_VALUE).put(Byte.MIN_VALUE).put((byte) 1).put((byte) -1);
        buf.rewind();
        str = Utils.toCharSequence(buf, DataType.INT8);
        Assert.assertEquals(str.toString(), " 127, -128,    1,   -1");

        buf.rewind();
        str = Utils.toCharSequence(buf, DataType.UINT8);
        Assert.assertEquals(str.toString(), "0x7F, 0x80, 0x01, 0xFF");

        buf.rewind();
        buf.limit(32);
        Utils.toCharSequence(buf, DataType.FLOAT16);
    }

    @Test
    public void testToFloatArray() {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(2);
        float[] array = Utils.toFloatArray(list);
        Assert.assertEquals(array, new float[] {1f, 2f});
    }
}
