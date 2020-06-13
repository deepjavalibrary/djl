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
import ai.djl.ModelException;
import ai.djl.fasttext.dataset.CookingStackExchange;
import ai.djl.fasttext.engine.FtModel;
import ai.djl.fasttext.engine.FtTrainingConfig;
import ai.djl.fasttext.engine.TextClassificationTranslator;
import ai.djl.fasttext.engine.Word2VecTranslator;
import ai.djl.fasttext.zoo.FtModelZoo;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrainCookingStackExchange {

    private static final Logger logger = LoggerFactory.getLogger(TrainCookingStackExchange.class);

    @Test
    public void testTrainTextClassification() throws IOException {
        try (FtModel model = (FtModel) Model.newInstance("ftModel")) {
            CookingStackExchange trainingSet = getDataset(Dataset.Usage.TRAIN);
            CookingStackExchange validateSet = getDataset(Dataset.Usage.TEST);

            // setup training configuration
            FtTrainingConfig config =
                    FtTrainingConfig.builder()
                            .setOutputDir(Paths.get("build"))
                            .setModelName("cooking")
                            .optLoss(FtTrainingConfig.FtLoss.HS)
                            .build();

            model.fit(config, trainingSet, validateSet);
            Assert.assertTrue(Files.exists(Paths.get("build/cooking.bin")));
        }
    }

    @Test
    public void testTextClassification()
            throws IOException, TranslateException, MalformedModelException,
                    ModelNotFoundException {
        try (ZooModel<String, Classifications> model =
                FtModelZoo.COOKING_STACKEXCHANGE.loadModel()) {
            try (Predictor<String, Classifications> predictor = model.newPredictor()) {
                Classifications result =
                        predictor.predict("Which baking dish is best to bake a banana bread ?");

                Assert.assertEquals(result.item(0).getClassName(), "bread");
            }
        }
    }

    @Test
    public void testWord2Vec()
            throws IOException, TranslateException, MalformedModelException,
                    ModelNotFoundException {
        try (ZooModel<String, Classifications> model =
                FtModelZoo.COOKING_STACKEXCHANGE.loadModel()) {
            Word2VecTranslator translator = new Word2VecTranslator();
            try (Predictor<String, float[]> predictor = model.newPredictor(translator)) {
                float[] result = predictor.predict("bread");
                Assert.assertEquals(result.length, 100);
                Assert.assertEquals(result[0], 0.038162477, 0.001);
            }
        }
    }

    @Test(enabled = false)
    public void testBlazingText() throws IOException, ModelException, TranslateException {
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }

        URL url =
                new URL(
                        "https://djl-ai.s3.amazonaws.com/resources/test-models/blazingtext_classification.bin");
        Path path = Paths.get("build/tmp/model");
        Path modelFile = path.resolve("text_classification.bin");
        if (!Files.exists(modelFile)) {
            Files.createDirectories(path);
            try (InputStream is = url.openStream()) {
                Files.copy(is, modelFile);
            }
        }

        TextClassificationTranslator translator = new TextClassificationTranslator();
        try (Model model = Model.newInstance("text_classification")) {
            model.load(path);
            try (Predictor<String, Classifications> predictor = model.newPredictor(translator)) {
                Classifications result =
                        predictor.predict(
                                "Convair was an american aircraft manufacturing company which later expanded into rockets and spacecraft .");

                logger.info("{}", result);
                Assert.assertEquals(result.item(0).getClassName(), "Company");
            }
        }
    }

    private static CookingStackExchange getDataset(Dataset.Usage usage) throws IOException {
        CookingStackExchange dataset = CookingStackExchange.builder().optUsage(usage).build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }
}
