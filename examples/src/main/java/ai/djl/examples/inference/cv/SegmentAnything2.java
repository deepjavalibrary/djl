/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.modality.cv.translator.Sam2TranslatorFactory;
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

public final class SegmentAnything2 {

    private static final Logger logger = LoggerFactory.getLogger(SegmentAnything2.class);

    private SegmentAnything2() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = predict();
        logger.info("{}", detection);
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        String url =
                "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg";
        Sam2Input input = Sam2Input.newInstance(url, 500, 375);

        Criteria<Sam2Input, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Sam2Input.class, DetectedObjects.class)
                        .optModelUrls("djl://ai.djl.pytorch/sam2-hiera-tiny")
                        .optEngine("PyTorch")
                        .optDevice(Device.cpu()) // use sam2-hiera-tiny-gpu for GPU
                        .optTranslatorFactory(new Sam2TranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Sam2Input, DetectedObjects> model = criteria.loadModel();
                Predictor<Sam2Input, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detection = predictor.predict(input);
            showMask(input, detection);
            return detection;
        }
    }

    private static void showMask(Sam2Input input, DetectedObjects detection) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        Image img = input.getImage();
        img.drawBoundingBoxes(detection, 0.8f);
        img.drawMarks(input.getPoints());

        Path imagePath = outputDir.resolve("sam2.png");
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Segmentation result image has been saved in: {}", imagePath);
    }
}
