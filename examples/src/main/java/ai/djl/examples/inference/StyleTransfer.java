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
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.StyleTransferTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class StyleTransfer {

    private static final Logger logger = LoggerFactory.getLogger(StyleTransfer.class);

    private StyleTransfer() {}

    public enum Artist {
        CEZANNE,
        MONET,
        UKIYOE,
        VANGOGH
    }

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Artist artist = Artist.MONET;
        String imagePath = "src/test/resources/mountains.png";
        Image input = ImageFactory.getInstance().fromFile(Paths.get(imagePath));
        Image output = transfer(input, artist);

        if (output == null) {
            logger.info("This example only works for PyTorch Engine");
        } else {
            logger.info("Using PyTorch Engine. " + artist + " painting generated.");
            save(output, artist.toString(), "build/output/cyclegan/");
        }
    }

    public static Image transfer(Image image, Artist artist)
            throws IOException, ModelNotFoundException, MalformedModelException,
                    TranslateException {

        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        String modelName = "style_" + artist.toString().toLowerCase() + ".zip";
        String modelUrl =
                "https://mlrepo.djl.ai/model/cv/image_generation/ai/djl/pytorch/cyclegan/0.0.1/"
                        + modelName;

        Criteria<Image, Image> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_GENERATION)
                        .setTypes(Image.class, Image.class)
                        .optModelUrls(modelUrl)
                        .optProgress(new ProgressBar())
                        .optTranslatorFactory(new StyleTransferTranslatorFactory())
                        .build();

        try (ZooModel<Image, Image> model = criteria.loadModel();
                Predictor<Image, Image> styler = model.newPredictor()) {
            return styler.predict(image);
        }
    }

    public static void save(Image image, String name, String path) throws IOException {
        Path outputPath = Paths.get(path);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(name + ".png");
        image.save(Files.newOutputStream(imagePath), "png");
    }
}
