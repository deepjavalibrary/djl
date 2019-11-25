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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ObjectDetection {

    private static final Logger logger = LoggerFactory.getLogger(ObjectDetection.class);

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = new ObjectDetection().predict();
        logger.info("{}", detection);
    }

    public DetectedObjects predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("size", "512");
        criteria.put("backbone", "resnet50");
        criteria.put("flavor", "v1");
        criteria.put("dataset", "voc");

        try (ZooModel<BufferedImage, DetectedObjects> model =
                MxModelZoo.SSD.loadModel(criteria, new ProgressBar())) {
            try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                Path output = drawBoundingBox(img, detection);
                logger.info("Detected objects image has been saved in: {}", output);
                return detection;
            }
        }
    }

    private static Path drawBoundingBox(BufferedImage img, DetectedObjects detection)
            throws IOException {
        Path dir = Paths.get("build/output");
        Files.createDirectories(dir);

        ImageVisualization.drawBoundingBoxes(img, detection);

        Path file = dir.resolve("detected-dog_bike_car.png");
        // OpenJDK can't save jpg with alpha channel
        ImageIO.write(img, "png", file.toFile());
        return file;
    }
}
