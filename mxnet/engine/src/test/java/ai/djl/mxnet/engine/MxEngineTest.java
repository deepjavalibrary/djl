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
// CHECKSTYLE:OFF:AvoidStaticImport

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.mxnet.jna.LibUtils;
import ai.djl.mxnet.jna.MxnetLibrary;
import ai.djl.mxnet.test.MockMxnetLibrary;
import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxEngineTest extends PowerMockTestCase {

    @BeforeClass
    public void prepare() throws IOException {
        mockStatic(LibUtils.class);
        MxnetLibrary library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
        Files.deleteIfExists(Paths.get("build/tmp/A-symbol.json"));
        Files.deleteIfExists(Paths.get("build/tmp/A-0122.params"));
        Files.deleteIfExists(Paths.get("build/tmp/A-0001.params"));
    }

    @AfterClass
    public void postProcessing() throws IOException {
        Files.deleteIfExists(Paths.get("build/tmp/A-symbol.json"));
        Files.deleteIfExists(Paths.get("build/tmp/A-0122.params"));
        Files.deleteIfExists(Paths.get("build/tmp/A-0001.params"));
    }

    @Test
    public void testGetGpuMemory() {
        Engine engine = Engine.getEngine(MxEngine.ENGINE_NAME);
        MemoryUsage usage = engine.getGpuMemory(Device.gpu(0));
        Assert.assertEquals(usage.getUsed(), 100);
        Assert.assertEquals(usage.getMax(), 1000);
    }

    @Test
    public void testDefaultDevice() {
        Engine engine = Engine.getEngine(MxEngine.ENGINE_NAME);
        Assert.assertEquals(engine.defaultDevice(), Device.gpu());
    }

    @Test
    public void testGetVersion() {
        Engine engine = Engine.getEngine(MxEngine.ENGINE_NAME);
        Assert.assertEquals(engine.getVersion(), "1.5.0");
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
