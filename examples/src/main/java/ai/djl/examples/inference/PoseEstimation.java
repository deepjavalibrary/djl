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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.Joints;
import ai.djl.modality.cv.Rectangle;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An example of inference using a pose estimation model.
 *
 * <p>See this <a
 * href="https://github.com/awslabs/djl/blob/master/examples/docs/pose_estimation.md">doc</a> for
 * information about this example.
 */
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

        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilter("size", "512")
                        .optFilter("backbone", "resnet50")
                        .optFilter("flavor", "v1")
                        .optFilter("dataset", "voc")
                        .optProgress(new ProgressBar())
                        .build();

        DetectedObjects detectedObjects;
        try (ZooModel<BufferedImage, DetectedObjects> ssd = ModelZoo.loadModel(criteria)) {
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

        Criteria<BufferedImage, Joints> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.POSE_ESTIMATION)
                        .setTypes(BufferedImage.class, Joints.class)
                        .optFilter("backbone", "resnet18")
                        .optFilter("flavor", "v1b")
                        .optFilter("dataset", "imagenet")
                        .build();

        try (ZooModel<BufferedImage, Joints> pose = ModelZoo.loadModel(criteria)) {
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
