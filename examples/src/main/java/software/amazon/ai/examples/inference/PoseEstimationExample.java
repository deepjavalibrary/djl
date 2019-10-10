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
package software.amazon.ai.examples.inference;

import ai.djl.mxnet.zoo.ModelZoo;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import software.amazon.ai.examples.inference.util.AbstractExample;
import software.amazon.ai.examples.inference.util.Arguments;
import software.amazon.ai.examples.inference.util.LogUtils;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.modality.cv.DetectedObjects;
import software.amazon.ai.modality.cv.ImageVisualization;
import software.amazon.ai.modality.cv.Joints;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.zoo.ModelNotFoundException;
import software.amazon.ai.zoo.ZooModel;

public class PoseEstimationExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(PoseEstimationExample.class);

    public static void main(String[] args) {
        new PoseEstimationExample().runExample(args);
    }

    @Override
    protected List<Joints> predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelNotFoundException, TranslateException {
        /* Section SSD */
        Path imageFile = arguments.getImageFile();
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);
        int imageWidth = img.getWidth();
        int imageHeight = img.getHeight();

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("size", "512");
        criteria.put("backbone", "resnet50_v1");
        criteria.put("dataset", "voc");
        ZooModel<BufferedImage, DetectedObjects> ssd = ModelZoo.SSD.loadModel(criteria);

        DetectedObjects ssdResult;
        try (Predictor<BufferedImage, DetectedObjects> predictor = ssd.newPredictor()) {
            ssdResult = predictor.predict(img);
        }
        ssd.close();

        // Get the cropped image
        List<BufferedImage> filtered =
                ssdResult
                        .items()
                        .stream()
                        .filter(obj -> obj.getClassName().equals("person"))
                        .map(obj -> obj.getBoundingBox().getBounds())
                        .map(
                                rect ->
                                        img.getSubimage(
                                                (int) (rect.getX() * imageWidth),
                                                (int) (rect.getY() * imageHeight),
                                                (int) (rect.getWidth() * imageWidth),
                                                (int) (rect.getHeight() * imageHeight)))
                        .collect(Collectors.toList());

        if (filtered.isEmpty()) {
            logger.warn("No person found in image.");
            return Collections.emptyList();
        }

        /* Pose recognition */
        criteria = new ConcurrentHashMap<>();
        criteria.put("flavor", "v1b");
        criteria.put("backbone", "resnet18");
        criteria.put("dataset", "imagenet");

        ZooModel<BufferedImage, Joints> pose = ModelZoo.SIMPLE_POSE.loadModel(criteria);

        List<Joints> poseResult = new ArrayList<>();
        try (Predictor<BufferedImage, Joints> predictor = pose.newPredictor()) {
            for (BufferedImage segmentedImg : filtered) {
                poseResult.add(predictor.predict(segmentedImg));
            }
        }
        pose.close();

        drawJoints(
                filtered.get(0),
                new Joints(
                        poseResult
                                .get(0)
                                .getJoints()
                                .stream()
                                .filter(ele -> ele.getConfidence() > 0.2f)
                                .collect(Collectors.toList())),
                arguments.getLogDir());
        return poseResult;
    }

    private void drawJoints(BufferedImage img, Joints joints, String logDir) throws IOException {
        if (logDir == null) {
            return;
        }

        Path dir = Paths.get(logDir);
        Files.createDirectories(dir);

        ImageVisualization.drawJoints(img, joints);

        Path out = Paths.get(logDir, "joint.png");
        ImageIO.write(img, "png", out.toFile());
    }
}
