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
import ai.djl.huggingface.translator.ZeroShotImageClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class ZeroShotImageClassification {

    private static final Logger logger = LoggerFactory.getLogger(ZeroShotImageClassification.class);

    private ZeroShotImageClassification() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Classifications classifications = predict();
        logger.info("{}", classifications);
    }

    public static Classifications predict() throws IOException, ModelException, TranslateException {
        String url = "https://resources.djl.ai/images/000000039769.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);

        VisionLanguageInput input =
                new VisionLanguageInput(img, new String[] {"cat", "remote control"});

        // You can use src/main/python/trace_clip_vit.py to trace the model
        Criteria<VisionLanguageInput, Classifications> criteria =
                Criteria.builder()
                        .setTypes(VisionLanguageInput.class, Classifications.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/openai/clip-vit-large-patch14")
                        .optEngine("PyTorch")
                        .optDevice(Device.cpu()) // Only support CPU
                        .optTranslatorFactory(new ZeroShotImageClassificationTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<VisionLanguageInput, Classifications> model = criteria.loadModel();
                Predictor<VisionLanguageInput, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(input);
        }
    }
}
