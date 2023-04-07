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
package ai.djl.util.passthrough;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

public class PassthroughNDManagerTest {

    @Test
    public void testPassthrough() {
        try (PassthroughNDManager manager = new PassthroughNDManager(Engine.getInstance(), null)) {
            Assert.assertNotNull(manager.getEngine());
            Assert.assertNotNull(manager.defaultDevice());

            NDManager sub = manager.newSubManager();
            sub.close();
            sub = manager.newSubManager(Device.gpu(1));
            Assert.assertEquals(sub.getDevice().getDeviceId(), 1);
            Assert.assertTrue(manager.getManagedArrays().isEmpty());

            Shape shape = new Shape(1);
            manager.cap();
            NDArray array = manager.zeros(shape);
            array.toByteBuffer();
            manager.from(array);

            PassthroughNDArray pa = manager.create((Object) "test");
            Assert.assertThrows(pa::toByteBuffer);
            Assert.assertEquals(pa.getObject(), "test");
            Assert.assertThrows(() -> pa.intern(null));
            pa.close();

            Assert.assertEquals(manager.getName(), "PassthroughNDManager");
            Assert.assertTrue(manager.isOpen());
            Assert.assertNotNull(manager.getParentManager());
            manager.attachInternal(null, (AutoCloseable) null);
            manager.attachUncappedInternal(null, null);
            manager.tempAttachInternal(null, null, null);
            manager.detachInternal(null);

            Assert.assertThrows(() -> manager.create(new String[0], null, null));
            Assert.assertThrows(() -> manager.create(shape, null));
            Assert.assertThrows(() -> manager.createCSR(null, null, null, shape));
            Assert.assertThrows(() -> manager.createRowSparse(null, shape, null, shape));
            Assert.assertThrows(() -> manager.createCoo(null, null, shape));
            Assert.assertThrows(() -> manager.load(null));
            Assert.assertThrows(() -> manager.full(shape, 1, null));
            Assert.assertThrows(() -> manager.arange(1f, 1f, 1f, null));
            Assert.assertThrows(() -> manager.eye(1, 1, 1, null));
            Assert.assertThrows(() -> manager.linspace(1f, 1f, 1, true));
            Assert.assertThrows(() -> manager.randomInteger(1L, 1L, shape, null));
            Assert.assertThrows(() -> manager.randomPermutation(1L));
            Assert.assertThrows(() -> manager.randomUniform(1f, 1f, shape, null));
            Assert.assertThrows(() -> manager.randomNormal(1f, 1f, shape, null));
            Assert.assertThrows(() -> manager.truncatedNormal(1f, 1f, shape, null));
            Assert.assertThrows(() -> manager.randomMultinomial(1, null));
            Assert.assertThrows(() -> manager.randomMultinomial(1, null, null));
            Assert.assertThrows(() -> manager.sampleNormal(null, null));
            Assert.assertThrows(() -> manager.sampleNormal(null, null, shape));
            Assert.assertThrows(() -> manager.samplePoisson(null));
            Assert.assertThrows(() -> manager.samplePoisson(null, shape));
            Assert.assertThrows(() -> manager.sampleGamma(null, null));
            Assert.assertThrows(() -> manager.sampleGamma(null, null, shape));
            Assert.assertThrows(() -> manager.invoke(null, null, null, null));
            Assert.assertThrows(() -> manager.invoke(null, null, null));
        }
    }
}
