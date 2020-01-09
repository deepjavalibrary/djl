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

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.Joints;
import ai.djl.modality.cv.Rectangle;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class PoseEstimation {

    private static final Logger logger = LoggerFactory.getLogger(PoseEstimation.class);

    private PoseEstimation() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Joints joints = PoseEstimation.predict();
        logger.info("{}", joints);
    }

    public static Joints predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/pose_soccer.png");
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        BufferedImage person = predictPersonInImage(img);

        if (person == null) {
            logger.warn("No person found in image.");
            return null;
        }

        return predictJointsInPerson(person);
    }

    private static BufferedImage predictPersonInImage(BufferedImage img)
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("size", "512");
        criteria.put("backbone", "resnet50");
        criteria.put("flavor", "v1");
        criteria.put("dataset", "voc");

        DetectedObjects detectedObjects;
        try (ZooModel<BufferedImage, DetectedObjects> ssd =
                MxModelZoo.SSD.loadModel(criteria, new ProgressBar())) {

            try (Predictor<BufferedImage, DetectedObjects> predictor = ssd.newPredictor()) {
                detectedObjects = predictor.predict(img);
            }
        }

        List<DetectedObjects.DetectedObject> items = detectedObjects.items();
        for (DetectedObjects.DetectedObject item : items) {
            if ("person".equals(item.getClassName())) {
                Rectangle rect = item.getBoundingBox().getBounds();
                int width = img.getWidth();
                int height = img.getHeight();
                return img.getSubimage(
                        (int) (rect.getX() * width),
                        (int) (rect.getY() * height),
                        (int) (rect.getWidth() * width),
                        (int) (rect.getHeight() * height));
            }
        }
        return null;
    }

    private static Joints predictJointsInPerson(BufferedImage person)
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("flavor", "v1b");
        criteria.put("backbone", "resnet18");
        criteria.put("dataset", "imagenet");

        try (ZooModel<BufferedImage, Joints> pose = MxModelZoo.SIMPLE_POSE.loadModel(criteria)) {

            try (Predictor<BufferedImage, Joints> predictor = pose.newPredictor()) {
                Joints joints = predictor.predict(person);
                saveJointsImage(person, joints);
                return joints;
            }
        }
    }

    private static void saveJointsImage(BufferedImage img, Joints joints) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        ImageVisualization.drawJoints(img, joints);

        Path imagePath = outputDir.resolve("joints.png");
        // Must use png format because you can't save as jpg with an alpha channel
        ImageIO.write(img, "png", imagePath.toFile());
        logger.info("Pose image has been saved in: {}", imagePath);
    }
}
