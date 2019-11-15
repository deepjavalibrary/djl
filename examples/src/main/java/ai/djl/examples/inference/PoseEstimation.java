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
import ai.djl.modality.cv.Joints;
import ai.djl.modality.cv.Rectangle;
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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PoseEstimation {

    private static final Logger logger = LoggerFactory.getLogger(PoseEstimation.class);

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Joints joints = new PoseEstimation().predict();
        logger.info("{}", joints);
    }

    public Joints predict() throws IOException, ModelException, TranslateException {
        /* Section SSD */
        Path imageFile = Paths.get("src/test/resources/pose_soccer.png");
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("size", "512");
        criteria.put("backbone", "resnet50");
        criteria.put("flavor", "v1");
        criteria.put("dataset", "voc");
        ZooModel<BufferedImage, DetectedObjects> ssd =
                MxModelZoo.SSD.loadModel(criteria, new ProgressBar());

        DetectedObjects ssdResult;
        try (Predictor<BufferedImage, DetectedObjects> predictor = ssd.newPredictor()) {
            ssdResult = predictor.predict(img);
        }
        ssd.close();

        BufferedImage person = null;
        List<DetectedObjects.DetectedObject> list = ssdResult.items();
        for (DetectedObjects.DetectedObject item : list) {
            if ("person".equals(item.getClassName())) {
                Rectangle rect = item.getBoundingBox().getBounds();
                int width = img.getWidth();
                int height = img.getHeight();
                person =
                        img.getSubimage(
                                (int) (rect.getX() * width),
                                (int) (rect.getY() * height),
                                (int) (rect.getWidth() * width),
                                (int) (rect.getHeight() * height));
                break;
            }
        }

        if (person == null) {
            logger.warn("No person found in image.");
            return null;
        }

        /* Pose recognition */
        criteria = new ConcurrentHashMap<>();
        criteria.put("flavor", "v1b");
        criteria.put("backbone", "resnet18");
        criteria.put("dataset", "imagenet");

        try (ZooModel<BufferedImage, Joints> pose = MxModelZoo.SIMPLE_POSE.loadModel(criteria)) {
            try (Predictor<BufferedImage, Joints> predictor = pose.newPredictor()) {
                Joints joints = predictor.predict(person);
                Path output = drawJoints(person, joints);
                logger.info("Pose image has been saved in: {}", output);
                return joints;
            }
        }
    }

    private static Path drawJoints(BufferedImage img, Joints joints) throws IOException {
        Path dir = Paths.get("build/output");
        Files.createDirectories(dir);

        ImageVisualization.drawJoints(img, joints);

        Path file = dir.resolve("joints.png");
        // OpenJDK can't save jpg with alpha channel
        ImageIO.write(img, "png", file.toFile());
        return file;
    }
}
