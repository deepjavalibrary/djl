/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloWorldTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class YoloWorld {

    private static final Logger logger = LoggerFactory.getLogger(YoloWorld.class);

    private YoloWorld() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = predict();
        logger.info("{}", detection);
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        String url = "https://resources.djl.ai/images/000000039769.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);
        VisionLanguageInput input =
                new VisionLanguageInput(img, new String[] {"cat", "remote control"});

        // You can use src/main/python/trace_yolo_worldv2.py to trace the model
        Criteria<VisionLanguageInput, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(VisionLanguageInput.class, DetectedObjects.class)
                        .optModelUrls("djl://ai.djl.pytorch/yolov8s-worldv2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new YoloWorldTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<VisionLanguageInput, DetectedObjects> model = criteria.loadModel();
                Predictor<VisionLanguageInput, DetectedObjects> predictor = model.newPredictor()) {
            Path outputPath = Paths.get("build/output");
            Files.createDirectories(outputPath);

            DetectedObjects detection = predictor.predict(input);
            if (detection.getNumberOfObjects() > 0) {
                img.drawBoundingBoxes(detection);
                Path output = outputPath.resolve("yolo_world.png");
                try (OutputStream os = Files.newOutputStream(output)) {
                    img.save(os, "png");
                }
                logger.info("Detected object saved in: {}", output);
            }
            return detection;
        }
    }
}
