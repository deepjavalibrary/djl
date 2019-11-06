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

package ai.djl.mxnet.engine;

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import ai.djl.mxnet.jna.LibUtils;
import ai.djl.mxnet.test.MockMxnetLibrary;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;

@PrepareForTest(LibUtils.class)
public class AutogradTest extends PowerMockTestCase {

    private static final Logger logger = LoggerFactory.getLogger(AutogradTest.class);
    private MockMxnetLibrary library;

    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
    }

    @Test
    public void testSetRecording() {
        logger.info("Test: Autograd setRecording");

        final int[] isRecording = new int[1];
        library.setFunction(
                "MXAutogradSetIsRecording",
                objects -> {
                    isRecording[0] = ((int) objects[0]);
                    ((IntBuffer) objects[1]).put(0, 0);
                    return 0;
                });
        boolean result = MxGradientCollector.setRecording(true);
        Assert.assertEquals(isRecording[0], 1);
        Assert.assertFalse(result);
    }

    @Test
    public void testSetTraining() {
        logger.info("Test: Autograd setTraining");
        final int[] isTraining = new int[1];
        library.setFunction(
                "MXAutogradSetIsTraining",
                objects -> {
                    isTraining[0] = ((int) objects[0]);
                    ((IntBuffer) objects[1]).put(0, 0);
                    return 0;
                });
        boolean result = MxGradientCollector.setTraining(true);
        Assert.assertEquals(isTraining[0], 1);
        Assert.assertFalse(result);
    }

    @Test
    public void testIsRecording() {
        logger.info("Test: Autograd isRecording");
        library.setFunction(
                "MXAutogradIsRecording",
                objects -> {
                    ((ByteBuffer) objects[0]).put(0, (byte) 1);
                    return 0;
                });
        boolean result = MxGradientCollector.isRecording();
        Assert.assertTrue(result);
    }

    @Test
    public void testIsTraining() {
        logger.info("Test: Autograd isTraining");
        library.setFunction(
                "MXAutogradIsTraining",
                objects -> {
                    ((ByteBuffer) objects[0]).put(0, (byte) 0);
                    return 0;
                });
        boolean result = MxGradientCollector.isTraining();
        Assert.assertFalse(result);
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
