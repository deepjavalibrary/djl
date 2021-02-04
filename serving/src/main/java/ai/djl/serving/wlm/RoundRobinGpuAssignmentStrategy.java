/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.wlm;

import ai.djl.serving.util.ConfigManager;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Assign next gpu using round robin to get the next gpuID.
 * 
 * @author erik.bamberg@web.de
 *
 */
public class RoundRobinGpuAssignmentStrategy implements GpuAssignmentStrategy {

    private static AtomicInteger gpuCounter=new AtomicInteger(0);

    private ConfigManager configManager;
    
    /**
     * Construct a round robin gpu assignment strategy.
     * 
     * @param configManager use this configuration manager.
     * 
     */
    public RoundRobinGpuAssignmentStrategy(ConfigManager configManager) {
	this.configManager=configManager;
    }
    
    /**
     * Returns next gpuId.
     * 
     * @return gpuId or -1 if no gpu is avaiable
     */
    @Override
    public int nextGpuId() {
        int gpuId = -1;
        int maxGpu = configManager.getNumberOfGpu();
        if (maxGpu > 0) {
            gpuId = gpuCounter.accumulateAndGet(maxGpu, (prev, maxGpuId) -> ++prev % maxGpuId);
        }
        return gpuId;
    }

}
