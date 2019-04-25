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
package org.apache.mxnet.jna;

import org.apache.mxnet.Context;
import org.testng.Assert;
import org.testng.annotations.Test;

public class JnaUtilsTest {

    @Test
    public void testGetVersion() {
        Assert.assertEquals(JnaUtils.getVersion(), 10500);
    }

    @Test
    public void testGetOpNames() {
        Assert.assertTrue(JnaUtils.getAllOpNames().size() >= 708);
    }

    @Test
    public void testGetNdArrayFunctions() {
        Assert.assertTrue(JnaUtils.getNdArrayFunctions().size() > 0);
    }

    @Test
    public void testGetGpuCount() {
        Assert.assertTrue(JnaUtils.getGpuCount() >= 0);
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testGetGpuMemoryIllegalArgument() {
        JnaUtils.getGpuMemory(Context.cpu());
    }

    @Test
    public void testGetGpuMemory() {
        if (JnaUtils.getGpuCount() > 0) {
            JnaUtils.getGpuMemory(Context.gpu(0));
        }
    }
}
