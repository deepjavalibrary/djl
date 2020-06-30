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
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.apache.commons.csv.CSVRecord;
import org.testng.Assert;
import org.testng.annotations.Test;

/*
 * 80 features
 *
 * Training Set:
 * 1460 Records
 *
 * Test Set:
 * 1459 Records
 *
 * Can enable/disable features
 * Set one hot vector for categorical variables
 */
public class AmesRandomAccessTest {

    @Test
    public void testAmesRandomAccessRemote() throws IOException, TranslateException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optInitializer(Initializer.ONES);

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            // path of directory
            AmesRandomAccess amesRandomAccess =
                    AmesRandomAccess.builder()
                            .optUsage(Dataset.Usage.TRAIN)
                            .setSampling(32, true)
                            .build();

            // Feature selection and options
            // Features start all enabled
            amesRandomAccess.removeAllFeatures();
            amesRandomAccess.addFeature("alley");
            amesRandomAccess.addFeature("MiscVal");
            amesRandomAccess.addFeature("id");
            amesRandomAccess.setOneHotEncode("alley", true); // 3 diff types (grvl, pave, na)

            amesRandomAccess.prepare();

            CSVRecord record0 = amesRandomAccess.getCSVRecord(0);
            CSVRecord record3 = amesRandomAccess.getCSVRecord(3);

            Assert.assertEquals(record0.get("LotShape"), "Reg");
            Assert.assertEquals(record3.get("LotShape"), "IR1");

            Assert.assertEquals(record0.get("YearBuilt"), "2003");
            Assert.assertEquals(record3.get("YearBuilt"), "1915");

            Assert.assertEquals(
                    amesRandomAccess.getFeatureNDArray(manager, 0).toFloatArray(),
                    new float[] {0, 1, 1, 0, 0});

            try (Trainer trainer = model.newTrainer(config)) {
                for (Batch batch : trainer.iterateDataset(amesRandomAccess)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                    batch.close();
                }
            }
        }
    }
}
