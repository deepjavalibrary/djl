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
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
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
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.util.Utils;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxModelTest extends PowerMockTestCase {

    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        MxnetLibrary library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
        Utils.deleteQuietly(Paths.get("build/tmp/testArt/"));
    }

    @AfterClass
    public void postProcess() {
        Utils.deleteQuietly(Paths.get("build/tmp/testArt/"));
    }

    @Test
    public void testLoadModel() throws IOException {
        String prefix = "A";
        int epoch = 122;
        try (MxModel model = MxModel.loadModel(prefix, epoch)) {
            Assert.assertEquals(
                    model.getBlock().getDirectParameters().get(0).getName(), "A-0122.params");
            Symbol sym = ((SymbolBlock) model.getBlock()).getSymbol();
            Assert.assertNotNull(sym);
        }
    }

    @Test
    public void testDescribeInput() throws IOException {
        String prefix = "A";
        int epoch = 122;
        try (MxModel model = MxModel.loadModel(prefix, epoch)) {
            DataDesc[] descs = model.describeInput();
            Assert.assertEquals(descs.length, 3);
            Assert.assertEquals(descs[0].getName(), "a");
        }
    }

    @Test
    public void testCast() throws IOException {
        String prefix = "A";
        int epoch = 122;
        MxModel model = MxModel.loadModel(prefix, epoch);
        MxModel casted = (MxModel) model.cast(DataType.FLOAT32);
        Assert.assertEquals(
                casted.getBlock().getDirectParameters(), model.getBlock().getDirectParameters());

        casted = (MxModel) model.cast(DataType.FLOAT64);
        Assert.assertEquals(
                casted.getBlock().getDirectParameters().get(0).getArray().getDataType(),
                DataType.FLOAT64);
        model.close();
    }

    @Test
    public void testGetArtifacts() throws IOException {
        String dir = "build/tmp/testArt/";
        String prefix = "A";
        int epoch = 122;
        // Test: Check filter
        Files.createDirectories(Paths.get(dir));
        Files.createFile(Paths.get(dir + prefix + "-0001.params"));
        Files.createFile(Paths.get(dir + prefix + "-symbol.json"));
        MxModel model = MxModel.loadModel(dir + prefix, epoch);
        Assert.assertEquals(model.getArtifactNames().length, 0);

        // Test: Add new file
        String synset = "synset.txt";
        Files.createFile(Paths.get(dir + synset));
        Assert.assertEquals(model.getArtifactNames()[0], synset);

        // Test: Add subDir
        Files.createDirectories(Paths.get(dir + "inner/"));
        Files.createFile(Paths.get(dir + "inner/innerFiles"));
        List<String> fileNames = Arrays.asList(model.getArtifactNames());
        Assert.assertTrue(fileNames.contains("inner/innerFiles"));

        // Test: Get Artifacts
        InputStream stream = model.getArtifactAsStream(synset);
        Assert.assertEquals(stream.available(), 0);
        Assert.assertNull(model.getArtifact("fileNotExist"));
        Assert.assertThrows(IllegalArgumentException.class, () -> model.getArtifact(null));
        // Test: Get Custom Artifacts
        Function<InputStream, String> wrongFunc =
                tempStream -> {
                    throw new RuntimeException("Test");
                };
        Assert.assertThrows(RuntimeException.class, () -> model.getArtifact(synset, wrongFunc));
        Function<InputStream, String> func = tempStream -> "Hello";
        String result = model.getArtifact(synset, func);
        Assert.assertEquals(result, "Hello");
        model.close();
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
