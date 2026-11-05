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
package ai.djl.pytorch.engine;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.IValue;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class PtIValueNDArrayTest {

    private static final String[] KEYS = {"a", "b"};
    private static final long[][] SHAPES = {{2}, {1, 3}};

    @BeforeClass
    public void loadEngine() {
        TestRequirements.notMacX86();
        Engine.getEngine("PyTorch");
    }

    private static PtIValueNDArray carrier(PtNDManager manager) {
        ByteBuffer parent = ByteBuffer.allocateDirect(20).order(ByteOrder.nativeOrder());
        ByteBuffer a = slice(parent, 0, 8);
        ByteBuffer b = slice(parent, 8, 12);
        return manager.createStringMap(
                "input",
                KEYS,
                new ByteBuffer[] {a, b},
                SHAPES,
                new DataType[] {DataType.FLOAT32, DataType.FLOAT32});
    }

    private static ByteBuffer slice(ByteBuffer parent, int offset, int length) {
        ByteBuffer dup = parent.duplicate();
        dup.position(offset);
        dup.limit(offset + length);
        return dup.slice().order(ByteOrder.nativeOrder());
    }

    @Test
    public void testMetadata() {
        TestRequirements.notMacX86();
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch")) {
            PtIValueNDArray c = carrier(manager);
            Assert.assertEquals(c.getName(), "input");
            Assert.assertEquals(c.getKeys(), KEYS);
            Assert.assertEquals(c.getShapes(), new Shape[] {new Shape(2), new Shape(1, 3)});
            Assert.assertTrue(c.getIValue().isMap());
            Assert.assertSame(c.getManager(), manager);
        }
    }

    @Test
    public void testUnsupportedOperationsThrow() {
        TestRequirements.notMacX86();
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch")) {
            PtIValueNDArray c = carrier(manager);
            // overridden operations name the carrier in the message
            Runnable[] ops = {c::getShape, c::getDataType, c::toByteBuffer, c::duplicate};
            for (Runnable op : ops) {
                try {
                    op.run();
                    Assert.fail("expected UnsupportedOperationException");
                } catch (UnsupportedOperationException e) {
                    Assert.assertTrue(e.getMessage().contains("PtIValueNDArray"), e.getMessage());
                }
            }
            // math operations fall through to the adapter's default
            Assert.assertThrows(UnsupportedOperationException.class, () -> c.add(1));
            Assert.assertThrows(UnsupportedOperationException.class, c::toFloatArray);
        }
    }

    @Test
    public void testClosedWithManager() {
        TestRequirements.notMacX86();
        PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch");
        PtIValueNDArray c = carrier(manager);
        manager.close();
        Assert.expectThrows(IllegalStateException.class, c::getIValue);
    }

    @Test
    public void testExplicitCloseIsIdempotent() {
        TestRequirements.notMacX86();
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch")) {
            PtIValueNDArray c = carrier(manager);
            c.close();
            c.close();
            Assert.expectThrows(IllegalStateException.class, c::getIValue);
        }
    }

    @Test
    public void testToStringDoesNotThrow() {
        TestRequirements.notMacX86();
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch")) {
            String str = carrier(manager).toString();
            Assert.assertTrue(str.contains("name=input"), str);
            for (String key : KEYS) {
                Assert.assertTrue(str.contains(key), str);
            }
        }
    }

    @Test
    public void testIdentityEqualsAndHashCode() {
        TestRequirements.notMacX86();
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch")) {
            PtIValueNDArray c1 = carrier(manager);
            PtIValueNDArray c2 = carrier(manager);
            Assert.assertTrue(c1.equals(c1));
            Assert.assertFalse(c1.equals(c2));
            Assert.assertFalse(c1.equals(null));
            Assert.assertEquals(c1.hashCode(), c1.hashCode());
            Assert.assertEquals(c1.hashCode(), System.identityHashCode(c1));
        }
    }

    @Test
    public void testShapesKeysLengthMismatch() {
        TestRequirements.notMacX86();
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch");
                IValue iv =
                        IValue.stringMapFrom(
                                new String[] {"a"},
                                new ByteBuffer[] {ByteBuffer.allocateDirect(8)},
                                new long[][] {{2}},
                                new DataType[] {DataType.FLOAT32})) {
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            new PtIValueNDArray(
                                    manager, "input", iv, new String[] {"a"}, new long[][] {}));
        }
    }

    @Test
    public void testNullOrEmptyName() {
        TestRequirements.notMacX86();
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch")) {
            ByteBuffer buf = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
            for (String name : new String[] {null, ""}) {
                Assert.assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                manager.createStringMap(
                                        name,
                                        new String[] {"a"},
                                        new ByteBuffer[] {buf},
                                        new long[][] {{2}},
                                        new DataType[] {DataType.FLOAT32}));
            }
        }
    }

    @Test
    public void testCreateOnClosedManagerThrows() {
        TestRequirements.notMacX86();
        PtNDManager manager = (PtNDManager) NDManager.newBaseManager("PyTorch");
        manager.close();
        // best effort: the IValue built before the failure must be closed, not leaked
        Assert.assertThrows(IllegalStateException.class, () -> carrier(manager));
    }
}
