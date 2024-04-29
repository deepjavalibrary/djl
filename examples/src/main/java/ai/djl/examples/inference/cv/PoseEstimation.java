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
package ai.djl.examples.inference.cv;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * An example of inference using a pose estimation model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/pose_estimation.md">doc</a>
 * for information about this example.
 */
public final class PoseEstimation {

    private static final Logger logger = LoggerFactory.getLogger(PoseEstimation.class);

    private PoseEstimation() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        List<Joints> joints = predict();
        logger.info("{}", joints);
    }

    public static List<Joints> predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/pose_soccer.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        List<Image> people = predictPeopleInImage(img);

        if (people.isEmpty()) {
            logger.warn("No people found in image.");
            return Collections.emptyList();
        }

        return predictJointsForPeople(people);
    }

    private static List<Image> predictPeopleInImage(Image img)
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelUrls("djl://ai.djl.mxnet/ssd/0.0.1/ssd_512_resnet50_v1_voc")
                        .optEngine("MXNet")
                        .optProgress(new ProgressBar())
                        .build();

        DetectedObjects detectedObjects;
        try (ZooModel<Image, DetectedObjects> ssd = criteria.loadModel();
                Predictor<Image, DetectedObjects> predictor = ssd.newPredictor()) {
            detectedObjects = predictor.predict(img);
        }

        List<DetectedObjects.DetectedObject> items = detectedObjects.items();
        List<Image> people = new ArrayList<>();
        for (DetectedObjects.DetectedObject item : items) {
            if ("person".equals(item.getClassName())) {
                Rectangle rect = item.getBoundingBox().getBounds();
                int width = img.getWidth();
                int height = img.getHeight();
                people.add(
                        img.getSubImage(
                                (int) (rect.getX() * width),
                                (int) (rect.getY() * height),
                                (int) (rect.getWidth() * width),
                                (int) (rect.getHeight() * height)));
            }
        }
        return people;
    }

    private static List<Joints> predictJointsForPeople(List<Image> people)
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {

        // Use DJL MXNet model zoo model, model can be found:
        // https://mlrepo.djl.ai/model/cv/pose_estimation/ai/djl/mxnet/simple_pose/0.0.1/simple_pose_resnet18_v1b-0000.params.gz
        // https://mlrepo.djl.ai/model/cv/pose_estimation/ai/djl/mxnet/simple_pose/0.0.1/simple_pose_resnet18_v1b-symbol.json
        Criteria<Image, Joints> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Joints.class)
                        .optModelUrls(
                                "djl://ai.djl.mxnet/simple_pose/0.0.1/simple_pose_resnet18_v1b")
                        .build();

        List<Joints> allJoints = new ArrayList<>();
        try (ZooModel<Image, Joints> pose = criteria.loadModel();
                Predictor<Image, Joints> predictor = pose.newPredictor()) {
            int count = 0;
            for (Image person : people) {
                Joints joints = predictor.predict(person);
                saveJointsImage(person, joints, count++);
                allJoints.add(joints);
            }
        }
        return allJoints;
    }

    private static void saveJointsImage(Image img, Joints joints, int count) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        img.drawJoints(joints);

        Path imagePath = outputDir.resolve("joints-" + count + ".png");
        // Must use png format because you can't save as jpg with an alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Pose image has been saved in: {}", imagePath);
    }
}
