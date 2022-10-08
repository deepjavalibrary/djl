/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Color;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * An example of inference using a semantic segmentation model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/semantic_segmentation.md">doc</a>
 * for information about this example.
 */
public final class SemanticSegmentation {

    private static final Logger logger = LoggerFactory.getLogger(SemanticSegmentation.class);

    private SemanticSegmentation() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        SemanticSegmentation.predict();
    }

    public static void predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String url =
                "https://mlrepo.djl.ai/model/cv/semantic_segmentation/ai/djl/pytorch/deeplabv3/0.0.1/deeplabv3.zip";

        Criteria<Image, CategoryMask> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.SEMANTIC_SEGMENTATION)
                        .setTypes(Image.class, CategoryMask.class)
                        .optModelUrls(url)
                        .optTranslatorFactory(new SemanticSegmentationTranslatorFactory())
                        .optEngine("PyTorch")
                        .optProgress(new ProgressBar())
                        .build();
        Image bg =
                ImageFactory.getInstance().fromFile(Paths.get("src/test/resources/airplane1.png"));
        try (ZooModel<Image, CategoryMask> model = criteria.loadModel();
                Predictor<Image, CategoryMask> predictor = model.newPredictor()) {
            CategoryMask mask = predictor.predict(img);

            // Highlights the detected object on the image with random opaque colors.
            Image img1 = img.duplicate();
            mask.drawMask(img1, 255);
            saveSemanticImage(img1, "semantic_instances1.png");

            // Highlights the detected object on the image with random colors.
            Image img2 = img.duplicate();
            mask.drawMask(img2, 100);
            saveSemanticImage(img2, "semantic_instances2.png");

            // Highlights the dog with blue color.
            Image img3 = img.duplicate();
            mask.drawMask(img3, 12, Color.BLUE.getRGB(), 100);
            saveSemanticImage(img3, "semantic_instances3.png");

            // Remove background
            Image maskImage = mask.getMaskImage(img);
            saveSemanticImage(maskImage, "semantic_instances4.png");

            // Replace background with an image
            bg = bg.resize(img.getWidth(), img.getHeight(), true);
            bg.drawImage(maskImage, true);
            saveSemanticImage(bg, "semantic_instances5.png");
        }
    }

    private static void saveSemanticImage(Image img, String fileName) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        Path imagePath = outputDir.resolve(fileName);
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Segmentation result image has been saved in: {}", imagePath.toAbsolutePath());
    }
}
