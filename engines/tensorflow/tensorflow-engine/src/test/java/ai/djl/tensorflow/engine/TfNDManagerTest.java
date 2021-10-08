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
package ai.djl.tensorflow.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TfNDManagerTest {

    @Test
    public void testNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create("string");
            Assert.assertEquals(array.toStringArray()[0], "string");

            array = manager.create(new String[] {"string1", "string2"});
            Assert.assertEquals(array.toStringArray()[1], "string2");
            final NDArray a = array;
            Assert.assertThrows(IllegalArgumentException.class, a::toByteBuffer);

            ByteBuffer buf1 = ByteBuffer.wrap("string1".getBytes(StandardCharsets.UTF_8));
            ByteBuffer buf2 = ByteBuffer.wrap(new byte[] {1, 0});

            array = ((TfNDManager) manager).createStringTensor(new Shape(2, 1), buf1, buf2);
            Assert.assertEquals(
                    array.toStringArray()[1].getBytes(StandardCharsets.UTF_8), buf2.array());

            array = manager.zeros(new Shape(2));
            final NDArray b = array;
            float[] expected = {2, 3};
            if (array.getDevice().isGpu()) {
                Assert.assertThrows(UnsupportedOperationException.class, () -> b.set(expected));
            } else {
                array.set(expected);
                Assert.assertEquals(array.toFloatArray(), expected);
            }

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> manager.create(new Shape(1), DataType.STRING));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> manager.create(buf1, new Shape(1), DataType.STRING));
        }
    }
}
