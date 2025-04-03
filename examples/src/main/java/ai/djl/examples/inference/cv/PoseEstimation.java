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

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.translator.YoloPoseTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

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
        Joints[] joints = predict();
        logger.info("{}", Arrays.toString(joints));
    }

    public static Joints[] predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/pose_soccer.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        // Use DJL PyTorch model zoo model
        Criteria<Image, Joints[]> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Joints[].class)
                        .optModelUrls("djl://ai.djl.pytorch/yolo11n-pose")
                        .optTranslatorFactory(new YoloPoseTranslatorFactory())
                        .build();

        try (ZooModel<Image, Joints[]> pose = criteria.loadModel();
                Predictor<Image, Joints[]> predictor = pose.newPredictor()) {
            Joints[] allJoints = predictor.predict(img);
            saveJointsImage(img, allJoints);
            return allJoints;
        }
    }

    private static void saveJointsImage(Image img, Joints[] allJoints) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        for (Joints joints : allJoints) {
            img.drawJoints(joints);
        }

        Path imagePath = outputDir.resolve("joints.png");
        // Must use png format because you can't save as jpg with an alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Pose image has been saved in: {}", imagePath);
    }
}
