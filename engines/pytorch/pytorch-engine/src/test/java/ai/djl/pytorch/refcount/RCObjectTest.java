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
package ai.djl.pytorch.refcount;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.refcount.RCObject;

import org.testng.Assert;
import org.testng.annotations.Test;

public class RCObjectTest {

    @Test
    public void testNDArraySimpleLifecycle() {
        System.out.println("NDArray simple lifecycle");
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new int[] {1, 2, 3});
            RCObject rco1 = (RCObject) array1;
            Assert.assertEquals(rco1.referenceCount(), 0);
            rco1.retainReference();
            Assert.assertEquals(rco1.referenceCount(), 1);
            Assert.assertTrue(rco1.releaseReference());
            Assert.assertEquals(rco1.referenceCount(), -1);
        }
    }

    @Test
    public void testNDArraySimpleLifecycle2() {
        System.out.println("NDArray simple lifecycle 2");
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.create(new int[] {1, 2, 3});
            RCObject rco1 = (RCObject) array1;
            Assert.assertEquals(rco1.referenceCount(), 0);
            rco1.retainReference();
            Assert.assertEquals(rco1.referenceCount(), 1);
            rco1.retainReference();
            Assert.assertEquals(rco1.referenceCount(), 2);
            Assert.assertFalse(rco1.releaseReference());
            Assert.assertEquals(rco1.referenceCount(), 1);
            Assert.assertTrue(rco1.releaseReference());
            Assert.assertEquals(rco1.referenceCount(), -1);
        }
    }
}
