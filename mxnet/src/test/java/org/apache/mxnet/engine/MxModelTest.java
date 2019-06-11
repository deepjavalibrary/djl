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

import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.test.MockMxnetLibrary;
import com.amazon.ai.util.Pair;
import com.amazon.ai.util.PairList;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.commons.io.FileUtils;
import org.apache.mxnet.jna.LibUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxModelTest extends PowerMockTestCase {
    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        MxnetLibrary library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
        FileUtils.deleteQuietly(new File("build/tmp/testArt/"));
    }

    @AfterClass
    public void postProcess() {
        FileUtils.deleteQuietly(new File("build/tmp/testArt/"));
    }

    @Test
    public void testLoadModel() throws IOException {
        String prefix = "A";
        int epoch = 122;
        MxModel model = MxModel.loadModel(prefix, epoch);
        Assert.assertEquals(model.getParameters().get(0).getKey(), "A-0122.params");
    }

    @Test
    public void testDescribeInput() throws IOException {
        String prefix = "A";
        int epoch = 122;
        MxModel model = MxModel.loadModel(prefix, epoch);
        PairList<String, MxNDArray> pairs = model.getParameters();
        pairs.remove("A-0122.params");
        pairs.add(new Pair<>("a", null));
        DataDesc[] descs = model.describeInput();
        // Comparing between a, b, c to a, b, c, d, e
        Assert.assertEquals(descs[0].getName(), "d");
        Assert.assertEquals(descs[1].getName(), "e");
    }

    @Test
    public void testGetArtifactNames() throws IOException {
        String dir = "build/tmp/testArt/";
        String prefix = "A";
        int epoch = 122;
        // Test: Check filter
        FileUtils.forceMkdir(new File(dir));
        Files.createFile(Paths.get(dir + prefix + "-0001.params"));
        MxModel model = MxModel.loadModel(dir + prefix, epoch);
        Assert.assertEquals(model.getArtifactNames().length, 0);
        // Test: Add new file
        String synset = "synset.txt";
        Files.createFile(Paths.get(dir + synset));
        Assert.assertEquals(model.getArtifactNames()[0], synset);
        // Test: add subDir
        FileUtils.forceMkdir(new File(dir + "inner/"));
        Files.createFile(Paths.get(dir + "inner/" + "innerFiles"));
        Assert.assertEquals(model.getArtifactNames()[0], "inner/innerFiles");
    }
}
