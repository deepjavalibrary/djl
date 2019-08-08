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
package org.apache.mxnet.dataset;

import java.io.IOException;
import java.util.Iterator;
import org.apache.mxnet.jna.JnaUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.util.Pair;

public class MnistTest {

    @BeforeClass
    public void setup() {
        JnaUtils.setNumpyMode(true);
    }

    @AfterClass
    public void tearDown() {
        JnaUtils.setNumpyMode(false);
    }

    @Test
    public void testMnistLocal() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Repository repository = Repository.newInstance("test", "src/test/resources/repo");
            Mnist mnist = Mnist.newInstance(manager, repository);
            mnist.prepare();
            Iterator<Pair<NDList, NDList>> it = mnist.getData(Dataset.Usage.TRAIN, 32);

            Assert.assertTrue(it.hasNext());

            Pair<NDList, NDList> pair = it.next();

            Assert.assertEquals(pair.getKey().size(), 1);
        }
    }

    @Test
    public void testMnistRemote() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Mnist mnist = Mnist.newInstance(manager);
            mnist.prepare();
            Iterator<Pair<NDList, NDList>> it = mnist.getData(Dataset.Usage.TRAIN, 32);

            Assert.assertTrue(it.hasNext());

            Pair<NDList, NDList> pair = it.next();

            Assert.assertEquals(pair.getKey().size(), 1);
        }
    }
}
