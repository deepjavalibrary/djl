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
package ai.djl.pytorch.jni;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class JniUtilsTest {

    @Test
    public void testClearGpuCache() {
        TestRequirements.notMacX86();

        JniUtils.emptyCudaCache();
    }

    /**
     * {@code torchFromBlob} passes the direct buffer's address to {@code torch::from_blob}, which
     * records the pointer without checking that the buffer is large enough for the requested shape.
     * Reading the resulting tensor would then access memory past the end of the buffer. The JNI
     * layer now rejects an undersized buffer instead.
     *
     * <p>{@link NDManager#create(java.nio.Buffer, Shape, DataType)} already rejects this in Java,
     * so the buffer is handed to {@link JniUtils} directly in order to reach the native guard.
     */
    @Test
    public void testCreateNdRejectsUndersizedBuffer() {
        TestRequirements.notMacX86();

        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager()) {
            ByteBuffer data = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder());
            // 2^30 FLOAT32 elements need 4 GiB, which no 16-byte buffer can back.
            Shape shape = new Shape(1L << 30);
            Assert.assertThrows(
                    EngineException.class,
                    () ->
                            JniUtils.createNdFromByteBuffer(
                                    manager,
                                    data,
                                    shape,
                                    DataType.FLOAT32,
                                    SparseFormat.DENSE,
                                    Device.cpu()));
        }
    }

    /** {@code torchSet} reaches the same unchecked address, through a different native call. */
    @Test
    public void testSetRejectsUndersizedBuffer() {
        TestRequirements.notMacX86();

        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager()) {
            PtNDArray array = (PtNDArray) manager.zeros(new Shape(1000), DataType.FLOAT32);
            ByteBuffer data = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder());
            Assert.assertThrows(EngineException.class, () -> JniUtils.set(array, data));
        }
    }

    /** A buffer that exactly fits the requested shape must still be accepted. */
    @Test
    public void testCreateNdAcceptsExactlySizedBuffer() {
        TestRequirements.notMacX86();

        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager()) {
            ByteBuffer data = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder());
            data.asFloatBuffer().put(new float[] {1f, 2f, 3f, 4f});
            PtNDArray array =
                    JniUtils.createNdFromByteBuffer(
                            manager,
                            data,
                            new Shape(4),
                            DataType.FLOAT32,
                            SparseFormat.DENSE,
                            Device.cpu());
            Assert.assertEquals(array.toFloatArray(), new float[] {1f, 2f, 3f, 4f});
        }
    }
}
