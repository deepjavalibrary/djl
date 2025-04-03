/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloV8TranslatorFactory;
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

/** An example of inference using an yolov8 model. */
public final class YoloDetection {

    private static final Logger logger = LoggerFactory.getLogger(YoloDetection.class);

    private YoloDetection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = predict();
        logger.info("{}", detection);
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        Path imgPath = Paths.get("src/test/resources/yolov8_test.jpg");
        Image img = ImageFactory.getInstance().fromFile(imgPath);

        // Use DJL OnnxRuntime model zoo model, model can be found:
        // https://mlrepo.djl.ai/model/cv/object_detection/ai/djl/onnxruntime/yolo11n/0.0.1/yolo11n.zip
        Criteria<Path, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Path.class, DetectedObjects.class)
                        .optModelUrls("djl://ai.djl.onnxruntime/yolo11n")
                        .optEngine("OnnxRuntime")
                        .optArgument("width", 640)
                        .optArgument("height", 640)
                        .optArgument("resize", true)
                        .optArgument("toTensor", true)
                        .optArgument("applyRatio", true)
                        .optArgument("threshold", 0.6f)
                        // for performance optimization maxBox parameter can reduce number of
                        // considered boxes from 8400
                        .optArgument("maxBox", 1000)
                        .optTranslatorFactory(new YoloV8TranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Path, DetectedObjects> model = criteria.loadModel();
                Predictor<Path, DetectedObjects> predictor = model.newPredictor()) {
            Path outputPath = Paths.get("build/output");
            Files.createDirectories(outputPath);

            DetectedObjects detection = predictor.predict(imgPath);
            if (detection.getNumberOfObjects() > 0) {
                img.drawBoundingBoxes(detection);
                Path output = outputPath.resolve("yolo_detected.png");
                try (OutputStream os = Files.newOutputStream(output)) {
                    img.save(os, "png");
                }
                logger.info("Detected object saved in: {}", output);
            }
            return detection;
        }
    }
}
