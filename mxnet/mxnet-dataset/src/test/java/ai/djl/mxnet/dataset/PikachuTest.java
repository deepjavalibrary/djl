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
package ai.djl.mxnet.dataset;

import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Repository;
import ai.djl.training.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.NormalInitializer;
import java.io.IOException;
import java.util.Iterator;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PikachuTest {

    @Test
    public void testPikachuLocal() throws IOException {
        Repository repository = Repository.newInstance("test", "src/test/resources/repo");
        PikachuDetection pikachu =
                new PikachuDetection.Builder()
                        .setUsage(Dataset.Usage.TEST)
                        .optRepository(repository)
                        .setRandomSampling(1)
                        .build();
        pikachu.prepare();
        TrainingConfig config = new DefaultTrainingConfig(new NormalInitializer(0.01));
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.IDENTITY_BLOCK);
            try (Trainer trainer = model.newTrainer(config)) {
                Iterator<Batch> ds = trainer.iterateDataset(pikachu).iterator();
                Batch batch = ds.next();
                Assert.assertEquals(batch.getData().head().getShape(), new Shape(1, 3, 512, 512));
                Assert.assertEquals(batch.getLabels().head().getShape(), new Shape(1, 1, 5));
            }
        }
    }
}
