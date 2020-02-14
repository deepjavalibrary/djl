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
import java.lang.management.MemoryUsage;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CudaUtilsTest {

    private static final Logger logger = LoggerFactory.getLogger(CudaUtilsTest.class);

    @Test
    public void testCudaUtils() {
        if (!CudaUtils.hasCuda()) {
            return;
        }
        // Possible to have CUDA and not have a GPU.
        if (CudaUtils.getGpuCount() == 0) {
            return;
        }

        int cudaVersion = CudaUtils.getCudaVersion();
        String smVersion = CudaUtils.getComputeCapability(0);
        MemoryUsage memoryUsage = CudaUtils.getGpuMemory(Device.gpu());

        logger.info("CUDA runtime version: {}, sm: {}", cudaVersion, smVersion);
        logger.info("Memory usage: {}", memoryUsage);

        Assert.assertTrue(cudaVersion >= 9020, "cuda 9.2+ required.");

        List<String> supportedSm = Arrays.asList("37", "52", "60", "61", "70", "75");
        Assert.assertTrue(supportedSm.contains(smVersion), "Unsupported cuda sm: " + smVersion);
    }
}
