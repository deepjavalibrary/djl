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

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** An example of generation using BigGAN. */
public final class BigGAN {

    private static final Logger logger = LoggerFactory.getLogger(BigGAN.class);

    private BigGAN() {}

    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        Image[] generatedImages = BigGAN.generate();

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

        for (int i = 0; i < generatedImages.length; ++i) {
            Path imagePath = outputPath.resolve("image" + i + ".png");
            generatedImages[i].save(Files.newOutputStream(imagePath), "png");
        }
        logger.info("Generated images have been saved in: {}", outputPath);
    }

    public static Image[] generate() throws IOException, ModelException, TranslateException {
        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        Criteria<int[], Image[]> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_GENERATION)
                        .setTypes(int[].class, Image[].class)
                        .optFilter("size", "256")
                        .optArgument("truncation", 0.4f)
                        .optProgress(new ProgressBar())
                        .build();

        int[] input = {100, 207, 971, 970, 933};

        try (ZooModel<int[], Image[]> model = criteria.loadModel();
                Predictor<int[], Image[]> generator = model.newPredictor()) {
            return generator.predict(input);
        }
    }
}
