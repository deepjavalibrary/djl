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
import ai.djl.basicdataset.tabular.MovieLens100k;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.repository.Repository;
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
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MovieLens100kTest {

    @Test
    public void testMovieLens100kRemote() throws IOException, TranslateException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();

            Repository repository =
                    Repository.newInstance(
                            "testrepo",
                            Paths.get(
                                    "/Users/siddhave/workplace/DeepJavaLibraryCore/djl/basicdataset/src/test/resources/mlrepo"));

            MovieLens100k movieLens100k =
                    MovieLens100k.builder()
                            .optUsage(Dataset.Usage.TEST)
                            .optRepository(repository)
                            .addFeature("user_age")
                            .addFeature("user_gender")
                            .addFeature("user_occupation")
                            .addFeature("movie_genres")
                            .setSampling(32, true)
                            .build();

            movieLens100k.prepare();

            long size = movieLens100k.size();
            Assert.assertEquals(size, 9430);

            Record record = movieLens100k.get(manager, 23);
            NDList data = record.getData();
            NDList labels = record.getLabels();

            Assert.assertEquals(
                    data.head().toFloatArray(),
                    new float[] {
                        23.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                    });
            Assert.assertEquals(labels.head().toFloatArray(), new float[] {5.0f});

            try (Trainer trainer = model.newTrainer(config)) {
                Batch batch = trainer.iterateDataset(movieLens100k).iterator().next();
                Assert.assertEquals(batch.getData().size(), 1);
                Assert.assertEquals(batch.getLabels().size(), 1);
                batch.close();
            }
        }
    }
}
