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
import org.junit.Assert;
import org.junit.Test;
import software.amazon.ai.Model;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.Dataset;

public class CocoTest {
    @Test
    public void testCocoLocal() throws IOException {
        Repository repository = Repository.newInstance("test", "src/test/resources/repo");
        CocoDetection coco =
                new CocoDetection.Builder()
                        .setUsage(Dataset.Usage.TEST)
                        .optRepository(repository)
                        .setSampling(1)
                        .build();
        coco.prepare();
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.IDENTITY_BLOCK);
            try (Trainer trainer = model.newTrainer()) {
                Iterator<Batch> ds = trainer.iterateDataset(coco).iterator();
                Batch batch = ds.next();
                Assert.assertEquals(batch.getData().head().getShape(), new Shape(1, 426, 640, 3));
                Assert.assertEquals(batch.getLabels().head().getShape(), new Shape(1, 20, 5));
            }
        }
    }
}
