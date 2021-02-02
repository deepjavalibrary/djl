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
import ai.djl.basicdataset.cv.classification.ImageNet;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset.Usage;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ImageNetTest {

    // ImageNet requires running manual download so can't be automatically tested
    @Test(enabled = false)
    public void testImageNetLocal() throws IOException, TranslateException {
        Repository repository =
                Repository.newInstance(
                        "test", System.getProperty("user.home") + "/Desktop/testImagenet");
        ImageNet imagenet =
                ImageNet.builder()
                        .optUsage(Usage.VALIDATION)
                        .setRepository(repository)
                        .setSampling(32, true)
                        .build();

        try (Model model = Model.newInstance("model")) {
            TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

            try (Trainer trainer = model.newTrainer(config)) {
                Batch batch = trainer.iterateDataset(imagenet).iterator().next();
                Assert.assertEquals(batch.getData().size(), 1);
                Assert.assertEquals(batch.getLabels().size(), 1);
                batch.close();
            }
        }
    }
}
