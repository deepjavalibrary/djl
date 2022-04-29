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
import ai.djl.basicdataset.tabular.AirfoilRandomAccess;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
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
public class AirfoilRandomAccessTest {

    @Test
    public void testAirfoilRemote() throws IOException, TranslateException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            AirfoilRandomAccess airfoil =
                    AirfoilRandomAccess.builder()
                            .optUsage(Dataset.Usage.TRAIN)
                            .addFeature("chordlen")
                            .addFeature("freq")
                            .setSampling(32, true)
                            .build();

            try (Trainer trainer = model.newTrainer(config)) {
                Batch batch = trainer.iterateDataset(airfoil).iterator().next();
                Assert.assertEquals(batch.getData().size(), 1);
                Assert.assertEquals(batch.getLabels().size(), 1);
                batch.close();
            }

            float epsilon = (float) 1e-4;
            Record record = airfoil.get(manager, 0);
            NDList data = record.getData();
            NDList labels = record.getLabels();

            Assert.assertEquals(data.head().toFloatArray(), new float[] {0.3048f, 800f}, epsilon);
            Assert.assertEquals(labels.head().toFloatArray(), new float[] {126.201f}, epsilon);
        }
    }

    @Test
    public void testAirfoilRemotePreprocessing() throws IOException, TranslateException {
        AirfoilRandomAccess airfoil =
                AirfoilRandomAccess.builder()
                        .optUsage(Dataset.Usage.TRAIN)
                        .optNormalize(true)
                        .optLimit(1500)
                        .setSampling(10, true)
                        .build();

        airfoil.prepare();

        NDManager manager = NDManager.newBaseManager();

        Record record = airfoil.get(manager, 0);
        NDList data = record.getData();
        NDList labels = record.getLabels();

        float epsilon = (float) 1e-4;

        float[] expected = {-0.6603f, -1.1448f, 1.797f, 1.3109f, -0.6443f};
        Assert.assertEquals(data.head().toFloatArray(), expected, epsilon);
        Assert.assertEquals(labels.head().toFloatArray(), new float[] {0.1937f}, epsilon);
    }
}
