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
package ai.djl.ndarray;

import org.testng.Assert;
import org.testng.annotations.Test;

public class NDScopeTest {

    @Test
    @SuppressWarnings("try")
    public void testNDScope() {
        NDArray detached;
        NDArray inside;
        NDArray uninvolved;
        try (NDManager manager = NDManager.newBaseManager()) {
            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();
                try (NDScope ignore = new NDScope()) {
                    uninvolved = manager.create(new int[] {1});
                    uninvolved.close();
                    inside = manager.create(new int[] {1});
                    // not tracked by any NDScope, but still managed by NDManager
                    NDScope.unregister(inside);
                }

                detached = manager.create(new int[] {1});
                detached.detach(); // detached from NDManager and NDScope
            }

            Assert.assertFalse(inside.isReleased());
        }
        Assert.assertTrue(inside.isReleased());
        Assert.assertFalse(detached.isReleased());
        detached.close();
    }
}
