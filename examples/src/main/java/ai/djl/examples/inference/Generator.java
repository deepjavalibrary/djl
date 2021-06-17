/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.input.BigGANInput;
import ai.djl.modality.cv.input.ImageNetCategory;
import ai.djl.modality.cv.translator.BigGANTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** An example of generation using BigGAN. */
public final class Generator {

    private static final Logger logger = LoggerFactory.getLogger(Generator.class);

    private Generator() {}

    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        Image[] generatedImages = Generator.generate();

        if (generatedImages == null) {
            logger.info("This example only works for PyTorch Engine");
        } else {
            logger.info("Using PyTorch Engine. {} images generated.", generatedImages.length);
            saveImages(generatedImages);
        }
    }

    private static void saveImages(Image[] generatedImages) throws IOException {
        Path outputPath = Paths.get("build/output/gan/");
        Files.createDirectories(outputPath);

        for (int i = 0; i < generatedImages.length; i++) {
            Path imagePath = outputPath.resolve("image" + i + ".jpg");
            generatedImages[i].save(Files.newOutputStream(imagePath), "jpg");
        }
        logger.info("Generated images have been saved in: {}", outputPath);
    }

    public static Image[] generate() throws IOException, ModelException, TranslateException {
        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        String modelPath = "build/models/gan/";
        String modelName = "biggan-deep-256";

        DownloadUtils.download(
                "https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/gan/ai/djl/pytorch/biggan-deep/0.0.1/"
                        + modelName
                        + ".pt.gz",
                modelPath + modelName + ".pt",
                new ProgressBar());

        Criteria<BigGANInput, Image[]> criteria =
                Criteria.builder()
                        .setTypes(BigGANInput.class, Image[].class)
                        .optModelName(modelName)
                        .optModelPath(Paths.get(modelPath))
                        .optTranslator(new BigGANTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        BigGANInput input =
                BigGANInput.builder()
                        .setCategory(ImageNetCategory.of("cheeseburger"))
                        .optSampleSize(5)
                        .optTruncation(0.5f)
                        .build();

        try (ZooModel<BigGANInput, Image[]> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BigGANInput, Image[]> generator = model.newPredictor()) {
                return generator.predict(input);
            }
        }
    }
}
