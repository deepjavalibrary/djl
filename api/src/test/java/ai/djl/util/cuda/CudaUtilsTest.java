/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util.cuda;

import ai.djl.Device;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.management.MemoryUsage;

public class CudaUtilsTest {

    private static final Logger logger = LoggerFactory.getLogger(CudaUtilsTest.class);

    @Test
    public void testCudaUtils() {
        if (!CudaUtils.hasCuda()) {
            Assert.assertThrows(CudaUtils::getCudaVersionString);
            Assert.assertThrows(() -> CudaUtils.getComputeCapability(0));
            Assert.assertThrows(() -> CudaUtils.getGpuMemory(Device.gpu()));
            return;
        }
        // Possible to have CUDA and not have a GPU.
        if (CudaUtils.getGpuCount() == 0) {
            return;
        }

        String cudaVersion = CudaUtils.getCudaVersionString();
        String smVersion = CudaUtils.getComputeCapability(0);
        MemoryUsage memoryUsage = CudaUtils.getGpuMemory(Device.gpu());

        logger.info("CUDA runtime version: {}, sm: {}", cudaVersion, smVersion);
        logger.info("Memory usage: {}", memoryUsage);

        Assert.assertNotNull(cudaVersion);
        Assert.assertNotNull(smVersion);
    }

    @Test
    public void testCudaUtilsWithFolk() {
        System.setProperty("ai.djl.util.cuda.folk", "true");
        try {
            testCudaUtils();
        } finally {
            System.clearProperty("ai.djl.util.cuda.folk");
        }
    }
}
