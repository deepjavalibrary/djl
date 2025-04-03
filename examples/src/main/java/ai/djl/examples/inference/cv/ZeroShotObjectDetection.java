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

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.huggingface.translator.ZeroShotObjectDetectionTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.output.DetectedObjects;
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

public final class ZeroShotObjectDetection {

    private static final Logger logger = LoggerFactory.getLogger(ZeroShotObjectDetection.class);

    private ZeroShotObjectDetection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects classifications = predict();
        logger.info("{}", classifications);
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        String url = "https://resources.djl.ai/images/000000039769.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);
        VisionLanguageInput input = new VisionLanguageInput(img, new String[] {"a cat"});

        // You can use src/main/python/trace_owlv2.py to trace the model
        Criteria<VisionLanguageInput, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(VisionLanguageInput.class, DetectedObjects.class)
                        .optModelUrls("djl://ai.djl.huggingface.pytorch/google/owlv2-base-patch16")
                        .optEngine("PyTorch")
                        .optDevice(Device.cpu()) // Only support CPU
                        .optTranslatorFactory(new ZeroShotObjectDetectionTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<VisionLanguageInput, DetectedObjects> model = criteria.loadModel();
                Predictor<VisionLanguageInput, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects ret = predictor.predict(input);
            img.drawBoundingBoxes(ret);

            Path outputDir = Paths.get("build/output");
            Files.createDirectories(outputDir);
            Path imagePath = outputDir.resolve("zero_shot_object_detection.png");
            // OpenJDK can't save jpg with alpha channel
            img.save(Files.newOutputStream(imagePath), "png");
            return ret;
        }
    }
}
