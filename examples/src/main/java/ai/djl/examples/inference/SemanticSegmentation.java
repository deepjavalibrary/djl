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

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An example of inference using a semantic segmentation model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/semantic_segmentation.md">doc</a>
 * for information about this example.
 */
public final class SemanticSegmentation {

    private static final Logger logger =
            LoggerFactory.getLogger(SemanticSegmentationTranslator.class);

    private SemanticSegmentation() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        SemanticSegmentation.predict();
    }

    public static void predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        int height = img.getHeight();
        int width = img.getWidth();

        String url =
                "https://mlrepo.djl.ai/model/cv/semantic_segmentation/ai/djl/pytorch/deeplabv3/0.0.1/deeplabv3.zip";
        Map<String, String> arguments = new ConcurrentHashMap<>();
        arguments.put("toTensor", "true");
        arguments.put("normalize", "true");
        SemanticSegmentationTranslator translator =
                SemanticSegmentationTranslator.builder(arguments).build();

        Criteria<Image, Image> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Image.class)
                        .optModelUrls(url)
                        .optTranslator(translator)
                        .optEngine("PyTorch")
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<Image, Image> model = criteria.loadModel()) {
            try (Predictor<Image, Image> predictor = model.newPredictor()) {
                Image semanticImage = predictor.predict(img);
                saveSemanticImage(semanticImage, width, height);
            }
        }
    }

    private static void saveSemanticImage(Image img, int width, int height) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        Path imagePath = outputDir.resolve("semantic_instances.png");

        // reduce image down to original size
        Image resized = img.getSubImage(0, 0, width, height);
        resized.save(Files.newOutputStream(imagePath), "png");
        logger.info("Segmentation result image has been saved in: {}", imagePath);
    }
}
