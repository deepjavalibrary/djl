/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.integration;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class PtNDArrayTest {

    @Test
    public void testComplexTensorCreation() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            Shape shape = new Shape(2);
            Assert.assertEquals(
                    manager.zeros(shape, DataType.COMPLEX64).getDataType(), DataType.COMPLEX64);
            Assert.assertEquals(
                    manager.ones(shape, DataType.COMPLEX64).getDataType(), DataType.COMPLEX64);
            Assert.assertEquals(
                    manager.create(shape, DataType.COMPLEX64).getDataType(), DataType.COMPLEX64);
        }
    }

    @Test
    public void testComplexTensorFromBuffer() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            float[] components = {3f, 4f, 5f, -12f};
            ByteBuffer data = manager.allocateDirect(components.length * Float.BYTES);
            data.asFloatBuffer().put(components);
            NDArray array = manager.create(data, new Shape(2), DataType.COMPLEX64);
            Assert.assertEquals(array.getDataType(), DataType.COMPLEX64);
            Assert.assertEquals(array.toByteBuffer().asFloatBuffer(), FloatBuffer.wrap(components));
        }
    }

    @Test
    public void testUnsupportedTensorDataType() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            Shape shape = new Shape(1);
            Assert.assertThrows(
                    EngineException.class, () -> manager.zeros(shape, DataType.UNKNOWN));
            ByteBuffer data = manager.allocateDirect(Integer.BYTES);
            Assert.assertThrows(
                    EngineException.class, () -> manager.create(data, shape, DataType.UINT32));
            for (DataType type : Arrays.asList(DataType.STRING, DataType.UNKNOWN)) {
                Assert.assertThrows(EngineException.class, () -> manager.create(data, shape, type));
                Assert.assertEquals(data.limit(), Integer.BYTES);
            }
        }
    }

    @Test
    public void testStringTensor() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager()) {
            String[] str = {"a", "b", "c"};
            NDArray arr = manager.create(str);
            Assert.assertEquals(arr.toString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(true), Arrays.toString(str));

            Assert.assertThrows(UnsupportedOperationException.class, () -> arr.get(0));
        }
    }

    @Test
    public void testLargeTensor() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(10 * 2850, 18944), DataType.FLOAT32);
            Assert.assertThrows(EngineException.class, array::toByteArray);
        }
    }
}
