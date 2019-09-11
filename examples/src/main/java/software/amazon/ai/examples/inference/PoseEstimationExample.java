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
import org.apache.mxnet.zoo.ModelZoo;
import org.slf4j.Logger;
import software.amazon.ai.Device;
import software.amazon.ai.examples.inference.util.AbstractExample;
import software.amazon.ai.examples.inference.util.Arguments;
import software.amazon.ai.examples.inference.util.LogUtils;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.modality.cv.DetectedObject;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.modality.cv.Joint;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.zoo.ModelNotFoundException;
import software.amazon.ai.zoo.ZooModel;

public class PoseEstimationExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(PoseEstimationExample.class);

    public static void main(String[] args) {
        new PoseEstimationExample().runExample(args);
    }

    @Override
    protected List<List<Joint>> predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelNotFoundException, TranslateException {
        /* Section SSD */
        List<DetectedObject> ssdResult;
        List<List<Joint>> poseResult;
        Path imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(imageFile);
        int imageWidth = img.getWidth();
        int imageHeight = img.getHeight();

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("size", "512");
        criteria.put("backbone", "resnet50_v1");
        criteria.put("dataset", "voc");
        ZooModel<BufferedImage, List<DetectedObject>> ssd = ModelZoo.SSD.loadModel(criteria);

        criteria = new ConcurrentHashMap<>();
        criteria.put("flavor", "v1b");
        criteria.put("backbone", "resnet18");
        criteria.put("dataset", "imagenet");

        ZooModel<BufferedImage, List<Joint>> pose = ModelZoo.SIMPLE_POSE.loadModel(criteria);

        Device device = Device.defaultDevice();

        try (Predictor<BufferedImage, List<DetectedObject>> ssdPredictor =
                ssd.newPredictor(device)) {
            ssdResult = ssdPredictor.predict(img);
        }
        // Get the cropped image
        List<BufferedImage> filtered =
                ssdResult
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
        try (Predictor<BufferedImage, List<Joint>> posePredictor = pose.newPredictor(device)) {
            posePredictor.setMetrics(metrics); // Let predictor collect metrics
            poseResult = new ArrayList<>();
            for (BufferedImage segmentedImg : filtered) {
                poseResult.add(posePredictor.predict(segmentedImg));
            }
            collectMemoryInfo(metrics);
        }

        drawJoints(
                filtered.get(0),
                poseResult
                        .get(0)
                        .stream()
                        .filter(ele -> ele.getConfidence() > 0.2f)
                        .collect(Collectors.toList()),
                arguments.getLogDir());
        ssd.close();
        pose.close();
        return poseResult;
    }

    private void drawJoints(BufferedImage img, List<Joint> joints, String logDir)
            throws IOException {
        if (logDir == null) {
            return;
        }

        Path dir = Paths.get(logDir);
        Files.createDirectories(dir);

        Images.drawJoints(img, joints);

        Path out = Paths.get(logDir, "joint.png");
        ImageIO.write(img, "png", out.toFile());
    }
}
