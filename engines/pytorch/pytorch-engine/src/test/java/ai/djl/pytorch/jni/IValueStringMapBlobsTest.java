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
package ai.djl.pytorch.jni;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

public class IValueStringMapBlobsTest {

    @BeforeClass
    public void loadEngine() {
        TestRequirements.notMacX86();
        // makes sure the native library is loaded before the first JNI call
        Engine.getEngine("PyTorch");
    }

    private static ByteBuffer slice(ByteBuffer parent, int offset, int length) {
        ByteBuffer dup = parent.duplicate();
        dup.position(offset);
        dup.limit(offset + length);
        return dup.slice().order(ByteOrder.nativeOrder());
    }

    private static ByteBuffer direct(int size) {
        return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
    }

    @Test
    public void testMixedDtypeRoundTrip() {
        TestRequirements.notMacX86();
        ByteBuffer parent = direct(36);
        ByteBuffer f = slice(parent, 0, 8);
        f.asFloatBuffer().put(new float[] {1.5f, -2f});
        ByteBuffer i = slice(parent, 8, 24);
        i.asLongBuffer().put(new long[] {7L, -8L, 9L});
        ByteBuffer g = slice(parent, 32, 4);
        g.asFloatBuffer().put(3f);

        try (NDManager manager = NDManager.newBaseManager();
                IValue map =
                        IValue.stringMapFrom(
                                new String[] {"f", "i", "g"},
                                new ByteBuffer[] {f, i, g},
                                new long[][] {{2}, {3}, {1}},
                                new DataType[] {
                                    DataType.FLOAT32, DataType.INT64, DataType.FLOAT32
                                })) {
            Assert.assertTrue(map.isMap());
            Map<String, IValue> values = map.toIValueMap();
            try {
                Assert.assertEquals(values.size(), 3);
                PtNDArray fa = values.get("f").toTensor((PtNDManager) manager);
                PtNDArray ia = values.get("i").toTensor((PtNDManager) manager);
                PtNDArray ga = values.get("g").toTensor((PtNDManager) manager);
                Assert.assertEquals(fa.toFloatArray(), new float[] {1.5f, -2f});
                Assert.assertEquals(ia.toLongArray(), new long[] {7L, -8L, 9L});
                Assert.assertEquals(ga.toFloatArray(), new float[] {3f});
                Assert.assertEquals(fa.getDataType(), DataType.FLOAT32);
                Assert.assertEquals(ia.getDataType(), DataType.INT64);
                Assert.assertEquals(ga.getDataType(), DataType.FLOAT32);
            } finally {
                for (IValue v : values.values()) {
                    v.close();
                }
            }
        }
    }

    @Test
    public void testZeroLengthTensor() {
        TestRequirements.notMacX86();
        ByteBuffer parent = direct(8);
        ByteBuffer empty = slice(parent, 4, 0);
        try (NDManager manager = NDManager.newBaseManager();
                IValue map =
                        IValue.stringMapFrom(
                                new String[] {"e"},
                                new ByteBuffer[] {empty},
                                new long[][] {{0}},
                                new DataType[] {DataType.INT64})) {
            Map<String, IValue> values = map.toIValueMap();
            try {
                PtNDArray e = values.get("e").toTensor((PtNDManager) manager);
                Assert.assertEquals(e.getShape(), new Shape(0));
            } finally {
                for (IValue v : values.values()) {
                    v.close();
                }
            }
        }
    }

    private static WeakReference<ByteBuffer> createMap(IValue[] out) {
        ByteBuffer parent = direct(16);
        ByteBuffer s = slice(parent, 0, 16);
        out[0] =
                IValue.stringMapFrom(
                        new String[] {"a"},
                        new ByteBuffer[] {s},
                        new long[][] {{4}},
                        new DataType[] {DataType.FLOAT32});
        return new WeakReference<>(parent);
    }

    @Test
    public void testBuffersKeptAliveUntilClose() {
        TestRequirements.notMacX86();
        IValue[] holder = new IValue[1];
        WeakReference<ByteBuffer> ref = createMap(holder);
        for (int i = 0; i < 3; ++i) {
            System.gc();
        }
        Assert.assertNotNull(ref.get());
        holder[0].close();
        Assert.assertThrows(IllegalStateException.class, () -> holder[0].getHandle());
    }

    @Test
    public void testInvalidArguments() {
        TestRequirements.notMacX86();
        ByteBuffer ok = direct(16);
        ByteBuffer ok2 = direct(16);
        String[] k1 = {"a"};
        ByteBuffer[] b1 = {ok};
        long[][] s1 = {{2}};
        DataType[] d1 = {DataType.FLOAT32};

        // unequal length
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> IValue.stringMapFrom(new String[] {"a", "b"}, b1, s1, d1));
        // all empty
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        IValue.stringMapFrom(
                                new String[0], new ByteBuffer[0], new long[0][], new DataType[0]));
        // null key
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> IValue.stringMapFrom(new String[] {null}, b1, s1, d1));
        // duplicate key
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        IValue.stringMapFrom(
                                new String[] {"a", "a"},
                                new ByteBuffer[] {ok, ok2},
                                new long[][] {{2}, {2}},
                                new DataType[] {DataType.FLOAT32, DataType.FLOAT32}));
        // null buffer
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> IValue.stringMapFrom(k1, new ByteBuffer[] {null}, s1, d1));
        // heap buffer
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> IValue.stringMapFrom(k1, new ByteBuffer[] {ByteBuffer.allocate(16)}, s1, d1));
        // unsupported dtypes
        for (DataType dt : new DataType[] {DataType.UNKNOWN, DataType.STRING, DataType.BFLOAT16}) {
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> IValue.stringMapFrom(k1, b1, s1, new DataType[] {dt}));
        }
        // negative dimension
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> IValue.stringMapFrom(k1, b1, new long[][] {{-1}}, d1));
    }

    @Test
    public void testBufferTooSmall() {
        TestRequirements.notMacX86();
        ByteBuffer parent = direct(16);
        ByteBuffer small = slice(parent, 0, 8);
        EngineException e =
                Assert.expectThrows(
                        EngineException.class,
                        () ->
                                IValue.stringMapFrom(
                                        new String[] {"a"},
                                        new ByteBuffer[] {small},
                                        new long[][] {{4}},
                                        new DataType[] {DataType.FLOAT32}));
        Assert.assertTrue(e.getMessage().contains("Buffer is too small"), e.getMessage());
    }
}
