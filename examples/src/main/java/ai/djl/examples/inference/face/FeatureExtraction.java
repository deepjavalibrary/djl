/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.face;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.ImageFeatureExtractorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public final class FeatureExtraction {

    private static final Logger logger = LoggerFactory.getLogger(FeatureExtraction.class);

    private FeatureExtraction() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/kana1.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        float[] feature = FeatureExtraction.predict(img);
        if (feature != null) {
            logger.info(Arrays.toString(feature));
        }
    }

    public static float[] predict(Image img)
            throws IOException, ModelException, TranslateException {
        img.getWrappedImage();

        List<Float> mean =
                Arrays.asList(
                        127.5f / 255.0f,
                        127.5f / 255.0f,
                        127.5f / 255.0f,
                        128.0f / 255.0f,
                        128.0f / 255.0f,
                        128.0f / 255.0f);
        String normalize = mean.stream().map(Object::toString).collect(Collectors.joining(","));

        Criteria<Image, float[]> criteria =
                Criteria.builder()
                        .setTypes(Image.class, float[].class)
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/pytorch/face_feature.zip")
                        .optModelName("face_feature") // specify model file prefix
                        .optArgument("normalize", normalize)
                        .optTranslatorFactory(new ImageFeatureExtractorFactory())
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch") // Use PyTorch engine
                        .build();

        try (ZooModel<Image, float[]> model = criteria.loadModel()) {
            Predictor<Image, float[]> predictor = model.newPredictor();
            return predictor.predict(img);
        }
    }
}
