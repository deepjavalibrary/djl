/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.fasttext.dataset.CookingStackExchange;
import ai.djl.fasttext.engine.FtTrainer;
import ai.djl.fasttext.engine.FtTrainingConfig;
import ai.djl.fasttext.engine.FtTranslator;
import ai.djl.fasttext.zoo.FtModelZoo;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TrainCookingStackExchange {

    @Test
    public void testTrainTextClassification() throws IOException {
        try (Model model = Model.newInstance()) {
            CookingStackExchange trainingSet = getDataset(Dataset.Usage.TRAIN);
            CookingStackExchange validateSet = getDataset(Dataset.Usage.TEST);

            // setup training configuration
            FtTrainingConfig config =
                    FtTrainingConfig.builder()
                            .setOutputDir(Paths.get("build"))
                            .setModelName("cooking")
                            .optLoss(FtTrainingConfig.FtLoss.HS)
                            .build();

            try (FtTrainer trainer = (FtTrainer) model.newTrainer(config)) {
                trainer.fit(trainingSet, validateSet);
            }
            Assert.assertTrue(Files.exists(Paths.get("build/cooking.bin")));
        }
    }

    @Test
    public void testTextClassification()
            throws IOException, TranslateException, MalformedModelException,
                    ModelNotFoundException {
        try (ZooModel<String, Classifications> model =
                FtModelZoo.COOKING_STACKEXCHANGE.loadModel()) {
            FtTranslator translator = new FtTranslator(5);
            try (Predictor<String, Classifications> predictor = model.newPredictor(translator)) {
                Classifications result =
                        predictor.predict("Which baking dish is best to bake a banana bread ?");
                System.out.println(result);
            }
        }
    }

    private static CookingStackExchange getDataset(Dataset.Usage usage) throws IOException {
        CookingStackExchange dataset = CookingStackExchange.builder().optUsage(usage).build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }
}
