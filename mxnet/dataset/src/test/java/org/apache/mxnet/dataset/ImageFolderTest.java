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
import org.apache.mxnet.engine.MxImages;
import org.testng.annotations.Test;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.dataset.Record;

public class ImageFolderTest {

    @Test
    public void testImageFolder() throws FailedTestException, IOException {
        ImageFolder dataset =
                new ImageFolder.Builder()
                        .setRoot("src/test/resources/imagefolder")
                        .setSampling(1, false, false)
                        .build();
        try (Trainer<String, Integer, NDList> trainer =
                        Trainer.newInstance(Block.IDENTITY_BLOCK, dataset.defaultTranslator());
                NDManager manager = NDManager.newBaseManager()) {

            NDArray cat = MxImages.read(manager, "src/test/resources/imagefolder/cat/cat2.jpeg");
            NDArray dog = MxImages.read(manager, "src/test/resources/imagefolder/dog/puppy1.jpg");

            Iterator<Record> ds = trainer.trainDataset(dataset).iterator();

            Record catRecord = ds.next();
            Assertions.assertAlmostEquals(cat, catRecord.getData().head());
            Assertions.assertEquals(manager.create(new int[] {0}), catRecord.getLabels().head());
            catRecord.close();

            Record dogRecord = ds.next();
            Assertions.assertAlmostEquals(dog, dogRecord.getData().head());
            Assertions.assertEquals(manager.create(new int[] {1}), dogRecord.getLabels().head());
            dogRecord.close();
        }
    }
}
