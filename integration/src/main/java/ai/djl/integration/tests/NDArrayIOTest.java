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
package ai.djl.integration.tests;

import ai.djl.engine.Engine;
import ai.djl.mxnet.engine.MxEngine;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.IntStream;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayIOTest {

    @Test
    public void testNDArrayLoad() {
        try (NDManager manager = NDManager.newBaseManager()) {
            ((MxEngine) Engine.getInstance()).setNumpyMode(false);
            Path arraysDictPath =
                    Paths.get(NDArrayIOTest.class.getResource("/two_arrays_dict").toURI());
            Path arraysListPath =
                    Paths.get(NDArrayIOTest.class.getResource("/two_arrays_list").toURI());
            NDList arraysDict = manager.load(arraysDictPath);
            NDList arraysList = manager.load(arraysListPath);
            Assert.assertEquals(arraysDict.size(), 2);
            Assert.assertEquals(arraysDict.getWithTag(0).getKey(), "x");
            Assert.assertEquals(arraysDict.getWithTag(1).getKey(), "y");
            Assert.assertEquals(arraysList.size(), 2);
            Assert.assertNull(arraysList.getWithTag(0).getKey());
            Assert.assertNull(arraysList.getWithTag(1).getKey());
        } catch (URISyntaxException e) {
            throw new AssertionError("URI parsing failed for test resources.", e);
        } finally {
            ((MxEngine) Engine.getInstance()).setNumpyMode(true);
        }
    }

    @Test
    public void testNDArraySaveLoadDict() {
        try (NDManager manager = NDManager.newBaseManager()) {
            File tmpfileNames = File.createTempFile("ndarray_list", "bin");
            int size = 10;
            int arangeStop = 25;
            NDList ndList = new NDList(size);
            IntStream.range(0, size)
                    .forEach((int x) -> ndList.add("array " + x, manager.arange(arangeStop)));
            manager.save(tmpfileNames.toPath(), ndList);
            NDList readNdList = manager.load(tmpfileNames.toPath());
            Assert.assertEquals(ndList, readNdList);
            tmpfileNames.deleteOnExit();
        } catch (IOException e) {
            throw new AssertionError("IOException while creating temporary file.", e);
        }
    }

    @Test
    public void testNDArraySaveLoadList() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            File tmpfileNames = File.createTempFile("ndarray_list", "bin");
            int size = 10;
            int arangeStop = 25;
            NDList ndList = new NDList(size);
            IntStream.range(0, size).forEach((int x) -> ndList.add(manager.arange(arangeStop)));
            manager.save(tmpfileNames.toPath(), ndList);
            NDList readNdList = manager.load(tmpfileNames.toPath());
            Assert.assertEquals(ndList, readNdList);
            tmpfileNames.deleteOnExit();
        }
    }
}
