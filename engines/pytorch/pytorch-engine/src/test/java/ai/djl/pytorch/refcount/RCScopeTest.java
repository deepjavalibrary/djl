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

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.refcount.RCObject;
import ai.djl.ndarray.refcount.RCScope;
import ai.djl.pytorch.engine.PtNDArray;

import org.testng.Assert;
import org.testng.annotations.Test;

public class RCScopeTest {

    @Test
    public void testRCScopeBaseProperties() {
        System.out.println("RCScope base properties");

        try (NDManager manager = NDManager.newBaseManager()) {
            RCObject outside = (RCObject) manager.create(new int[] {1});
            RCObject attached = (RCObject) manager.create(new int[] {1});
            RCObject detached;
            RCObject inside;
            RCObject inside1;
            RCObject inside2;
            RCObject retained1;
            RCObject retained2;
            RCObject inside5;

            try (RCScope scope = new RCScope()) {
                scope.attach(attached);

                detached = (RCObject) manager.create(new int[] {1});
                detached.retainReference();
                scope.detach(detached);

                inside = (RCObject) manager.create(new int[] {1});
                try (RCScope scope1 = new RCScope()) {
                    scope1.suppressNotUsedWarning();
                    inside1 = (RCObject) manager.create(new int[] {1});
                    inside2 = (RCObject) manager.create(new int[] {1});
                }
                try (RCScope scope2 = new RCScope()) {
                    scope2.suppressNotUsedWarning();
                    retained1 = (RCObject) manager.create(new int[] {1});
                    retained2 = (RCObject) manager.create(new int[] {1});
                    retained1.retainReference();
                    scope.attach(retained2);
                }
                retained2.retainReference();
                inside5 = (RCObject) manager.create(new int[] {1});
            }

            RCObject outside2 = (RCObject) manager.create(new int[] {1});

            Assert.assertFalse(outside.isNull());
            Assert.assertTrue(attached.isNull());
            Assert.assertFalse(detached.isNull());
            Assert.assertTrue(inside.isNull());
            Assert.assertTrue(inside1.isNull());
            Assert.assertTrue(inside2.isNull());
            Assert.assertFalse(retained1.isNull());
            Assert.assertFalse(retained2.isNull());
            Assert.assertTrue(inside5.isNull());
            Assert.assertFalse(outside2.isNull());

            outside.releaseReference();
            detached.releaseReference();
            retained1.releaseReference();
            retained2.releaseReference();
            outside2.releaseReference();

            Assert.assertTrue(outside.isNull());
            Assert.assertTrue(detached.isNull());
            Assert.assertTrue(retained1.isNull());
            Assert.assertTrue(retained2.isNull());
            Assert.assertTrue(outside2.isNull());
        }
    }

    @Test
    public void testRCScopeDetachingFromManager() {
        System.out.println("RCScope detaching from manager");
        PtNDArray inside;

        try (NDManager manager = NDManager.newBaseManager()) {
            try (RCScope scope = new RCScope()) {
                scope.suppressNotUsedWarning();
                inside = (PtNDArray) manager.create(new int[] {1});
            }
            Assert.assertFalse(inside.getManager().hasResource(inside));
            Assert.assertFalse(inside.getManager().hasTempResource(inside));
        }
    }
}
