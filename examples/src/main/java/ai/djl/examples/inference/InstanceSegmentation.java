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
import java.awt.Graphics2D;
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

public class InstanceSegmentation {

    private static final Logger logger = LoggerFactory.getLogger(InstanceSegmentation.class);

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = new InstanceSegmentation().predict();
        logger.info("{}", detection);
    }

    public DetectedObjects predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/segmentation.jpg");
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("flavor", "v1b");
        criteria.put("backbone", "resnet18");
        criteria.put("dataset", "coco");

        try (ZooModel<BufferedImage, DetectedObjects> model =
                MxModelZoo.MASK_RCNN.loadModel(criteria, new ProgressBar())) {
            try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                Path output = drawBoundingBox(img, detection);
                logger.info("Segmentation result image has been saved in: {}", output);
                return detection;
            }
        }
    }

    private static Path drawBoundingBox(BufferedImage img, DetectedObjects detection)
            throws IOException {
        Path dir = Paths.get("build/output");
        Files.createDirectories(dir);
        // Make copy with alpha channel
        BufferedImage newImage =
                new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = newImage.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();
        ImageVisualization.drawBoundingBoxes(newImage, detection);

        Path file = dir.resolve("instances.png");
        ImageIO.write(newImage, "png", file.toFile());
        return file;
    }
}
