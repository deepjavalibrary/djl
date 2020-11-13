/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.integration.tests.ndarray;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDListTest {

    @Test
    public void testEncode() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Assert.assertThrows(
                    () -> {
                        // invalid NDList length
                        byte[] data = {-1, 0, 0, 0};
                        NDList.decode(manager, data);
                    });
            Assert.assertThrows(
                    () -> {
                        // large NDList size should not cause OOM
                        byte[] data = {0x7f, 0, 0, 0};
                        NDList.decode(manager, data);
                    });
            Assert.assertThrows(
                    () -> {
                        // Test invalid NDArray header
                        byte[] data = {0, 0, 0, 1, 0, 4, 78, 68, 65, 81};
                        NDList.decode(manager, data);
                    });
            Assert.assertThrows(
                    () -> {
                        // Test invalid NDArray version
                        byte[] data = {0, 0, 0, 1, 0, 4, 78, 68, 65, 82, -1, 0, 0, 0};
                        NDList.decode(manager, data);
                    });
            Assert.assertThrows(
                    () -> {
                        // corrupted NDArray data
                        byte[] data = {0, 0, 0, 1, 0, 4, 78, 68, 65, 82, 0, 0, 0, 2, 1, 5};
                        NDList.decode(manager, data);
                    });
        }
    }
}
