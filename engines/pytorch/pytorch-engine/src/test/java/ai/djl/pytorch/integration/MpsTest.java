/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class MpsTest {

    @Test
    public void testMps() {
        if (!"aarch64".equals(System.getProperty("os.arch"))
                || !System.getProperty("os.name").startsWith("Mac")) {
            throw new SkipException("MPS test requires M1 macOS.");
        }

        Device device = Device.of("mps", -1);
        try (NDManager manager = NDManager.newBaseManager(device)) {
            NDArray array = manager.zeros(new Shape(1, 2));
            Assert.assertEquals(array.getDevice().getDeviceType(), "mps");
        }
    }
}
