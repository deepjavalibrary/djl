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
package ai.djl.basicdataset;

import ai.djl.Model;
import ai.djl.basicdataset.cv.PikachuDetection;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Iterator;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PikachuTest {

    @Test
    public void testPikachuRemote() throws IOException, TranslateException {
        PikachuDetection pikachu =
                PikachuDetection.builder()
                        .optArtifactId("pikachu-unittest")
                        .optUsage(Dataset.Usage.TEST)
                        .setSampling(1, true)
                        .optLimit(10)
                        .build();
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optInitializer(new NormalInitializer(0.01f), Parameter.Type.WEIGHT);
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());
            try (Trainer trainer = model.newTrainer(config)) {
                Iterator<Batch> ds = trainer.iterateDataset(pikachu).iterator();
                Batch batch = ds.next();
                Assert.assertEquals(
                        batch.getData().singletonOrThrow().getShape(), new Shape(1, 3, 256, 256));
                Assert.assertEquals(
                        batch.getLabels().singletonOrThrow().getShape(), new Shape(1, 1, 5));
            }
        }
    }
}
