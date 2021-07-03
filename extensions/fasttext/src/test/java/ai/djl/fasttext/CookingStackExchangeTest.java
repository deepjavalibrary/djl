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
import ai.djl.ModelException;
import ai.djl.basicdataset.nlp.CookingStackExchange;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.TrainingResult;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class CookingStackExchangeTest {

    private static final Logger logger = LoggerFactory.getLogger(CookingStackExchangeTest.class);

    @Test
    public void testTrainTextClassification() throws IOException {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            throw new SkipException("fastText is not supported on windows");
        }
        try (FtModel model = new FtModel("cooking")) {
            CookingStackExchange dataset = CookingStackExchange.builder().build();

            // setup training configuration
            FtTrainingConfig config =
                    FtTrainingConfig.builder()
                            .setOutputDir(Paths.get("build"))
                            .setModelName("cooking")
                            .optEpoch(5)
                            .optLoss(FtTrainingConfig.FtLoss.HS)
                            .build();

            TrainingResult result = model.fit(config, dataset);
            Assert.assertEquals(result.getEpoch(), 5);
            Assert.assertTrue(Files.exists(Paths.get("build/cooking.bin")));
        }
    }

    @Test
    public void testTextClassification()
            throws IOException, MalformedModelException, ModelNotFoundException {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            throw new SkipException("fastText is not supported on windows");
        }
        Criteria<String, Classifications> criteria =
                Criteria.builder()
                        .setTypes(String.class, Classifications.class)
                        .optArtifactId("ai.djl.fasttext:cooking_stackexchange")
                        .build();
        try (ZooModel<String, Classifications> model = criteria.loadModel()) {
            String input = "Which baking dish is best to bake a banana bread ?";
            FtModel ftModel = (FtModel) model.getWrappedModel();
            Classifications result = ftModel.classify(input, 8);
            Assert.assertEquals(result.item(0).getClassName(), "bread");
        }
    }

    @Test
    public void testWord2Vec() throws IOException, MalformedModelException, ModelNotFoundException {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            throw new SkipException("fastText is not supported on windows");
        }
        Criteria<String, Classifications> criteria =
                Criteria.builder()
                        .setTypes(String.class, Classifications.class)
                        .optArtifactId("ai.djl.fasttext:cooking_stackexchange")
                        .build();
        try (ZooModel<String, Classifications> model = criteria.loadModel();
                NDManager manager = NDManager.newBaseManager()) {

            FtModel ftModel = (FtModel) model.getWrappedModel();
            FtWord2VecWordEmbedding fasttextWord2VecWordEmbedding =
                    new FtWord2VecWordEmbedding(ftModel, new FtVocabulary());
            long index = fasttextWord2VecWordEmbedding.preprocessWordToEmbed("bread");
            NDArray embedding = fasttextWord2VecWordEmbedding.embedWord(manager, index);
            Assert.assertEquals(embedding.getShape(), new Shape(100));
            Assert.assertEquals(embedding.toFloatArray()[0], 0.038162477, 0.001);
        }
    }

    @Test
    public void testBlazingText() throws IOException, ModelException {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            throw new SkipException("fastText is not supported on windows");
        }
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }

        URL url = new URL("https://resources.djl.ai/test-models/blazingtext_classification.bin");
        Path path = Paths.get("build/tmp/model");
        Path modelFile = path.resolve("text_classification.bin");
        if (!Files.exists(modelFile)) {
            Files.createDirectories(path);
            try (InputStream is = url.openStream()) {
                Files.copy(is, modelFile, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        try (FtModel model = new FtModel("text_classification")) {
            model.load(modelFile);
            String text =
                    "Convair was an american aircraft manufacturing company which later expanded into rockets and spacecraft .";
            Classifications result = model.classify(text, 5);

            logger.info("{}", result);
            Assert.assertEquals(result.item(0).getClassName(), "Company");
        }
    }
}
