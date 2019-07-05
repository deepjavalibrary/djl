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
package org.apache.mxnet.engine;
// CHECKSTYLE:OFF:AvoidStaticImport

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.jna.LibUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.apache.mxnet.test.MockMxnetLibrary;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;
import software.amazon.ai.Context;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxEngineTest extends PowerMockTestCase {

    @BeforeClass
    public void prepare() throws IOException {
        mockStatic(LibUtils.class);
        MxnetLibrary library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
        Files.deleteIfExists(Paths.get("build/tmp/A-0122.params"));
        Files.deleteIfExists(Paths.get("build/tmp/A-0001.params"));
    }

    @AfterClass
    public void postProcessing() throws IOException {
        Files.deleteIfExists(Paths.get("build/tmp/A-0122.params"));
        Files.deleteIfExists(Paths.get("build/tmp/A-0001.params"));
    }

    @Test
    public void testGetGpuMemory() {
        MxEngine engine = new MxEngine();
        MemoryUsage usage = engine.getGpuMemory(Context.gpu(0));
        Assert.assertEquals(usage.getUsed(), 100);
        Assert.assertEquals(usage.getMax(), 1000);
    }

    @Test
    public void testDefaultContext() {
        MxEngine engine = new MxEngine();
        Assert.assertEquals(engine.defaultContext(), Context.gpu());
    }

    @Test
    public void testGetVersion() {
        MxEngine engine = new MxEngine();
        Assert.assertEquals(engine.getVersion(), "1.5.0");
    }

    @Test
    public void testLoadModel() throws IOException {
        MxEngine engine = new MxEngine();
        Path modelDir = Paths.get("build/tmp/");
        String modelName = "A";
        Path path122 = modelDir.resolve(modelName + "-0122.params");
        Files.createFile(path122);
        String result = loadModel(engine, modelDir, modelName, null);
        Files.delete(path122);
        Assert.assertEquals(result, "A-0122.params");

        Path path1 = modelDir.resolve(modelName + "-0001.params");
        Files.createFile(path1);
        result = loadModel(engine, modelDir, modelName, null);
        Files.delete(path1);
        Assert.assertEquals(result, "A-0001.params");

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("epoch", String.valueOf(122));
        result = loadModel(engine, modelDir, modelName, options);
        Assert.assertEquals(result, "A-0122.params");

        Files.createFile(path122);
        Files.createFile(path1);

        result = loadModel(engine, modelDir, modelName, options);
        Assert.assertEquals(result, "A-0122.params");
    }

    private String loadModel(
            MxEngine engine, Path location, String modelName, Map<String, String> options)
            throws IOException {
        try (MxModel model = (MxModel) engine.loadModel(location, modelName, options)) {
            // In JNA.MXNDArrayLoad function, file name is stored as the first param name in Model
            String paramPath = model.getParameters().get(0).getKey();
            return Paths.get(paramPath).toFile().getName();
        }
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
