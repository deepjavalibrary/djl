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
import java.nio.file.Paths;
import java.util.Iterator;
import org.testng.annotations.Test;
import software.amazon.ai.Model;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.modality.cv.util.NDImageUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.initializer.Initializer;

public class ImageFolderTest {

    @Test
    public void testImageFolder() throws IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.IDENTITY_BLOCK);

            ImageFolder dataset =
                    new ImageFolder.Builder()
                            .setManager(model.getNDManager())
                            .setRoot("src/test/resources/imagefolder")
                            .setResize(100, 100)
                            .setSequenceSampling(1, false)
                            .build();
            dataset.prepare();

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray cat =
                        BufferedImageUtils.readFileToArray(
                                manager, Paths.get("src/test/resources/imagefolder/cat/cat2.jpeg"));
                NDArray dog =
                        BufferedImageUtils.readFileToArray(
                                manager,
                                Paths.get("src/test/resources/imagefolder/dog/puppy1.jpg"));

                Iterator<Batch> ds = trainer.iterateDataset(dataset).iterator();

                Batch catBatch = ds.next();
                Assertions.assertAlmostEquals(
                        NDImageUtils.resize(cat, 100, 100).expandDims(0),
                        catBatch.getData().head());
                Assertions.assertEquals(manager.create(new int[] {0}), catBatch.getLabels().head());
                catBatch.close();

                Batch dogBatch = ds.next();
                Assertions.assertAlmostEquals(
                        NDImageUtils.resize(dog, 100, 100).expandDims(0),
                        dogBatch.getData().head());
                Assertions.assertEquals(manager.create(new int[] {1}), dogBatch.getLabels().head());
                dogBatch.close();
            }
        }
    }
}
