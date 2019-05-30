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

// CHECKSTYLE:OFF:AvoidStaticImport
import static org.powermock.api.mockito.PowerMockito.mockStatic;

import com.amazon.ai.Context;
import java.nio.IntBuffer;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class JnaUtilsTest extends PowerMockTestCase {

    @Test
    public void testGetVersion() {
        mockStatic(LibUtils.class);

        IntBuffer buf = IntBuffer.allocate(1);
        MxnetLibrary library = Mockito.mock(MxnetLibrary.class);
        PowerMockito.when(library.MXGetVersion(buf))
                .thenAnswer(
                        (Answer<Integer>)
                                invocation -> {
                                    Object[] args = invocation.getArguments();
                                    ((IntBuffer) args[0]).put(0, 10500);
                                    return 0;
                                });

        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);

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

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
