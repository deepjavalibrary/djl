/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.Segmentation;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * An example of inference using a semantic segmentation model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/semantic_segmentation.md">doc</a>
 * for information about this example.
 */
public final class SemanticSegmentation {

    private static final Logger logger =
            LoggerFactory.getLogger(SemanticSegmentationTranslator.class);

    private SemanticSegmentation() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        logger.info("Result: {}", predict().getMask().length);
    }

    public static Segmentation predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        String url =
                "https://mlrepo.djl.ai/model/cv/semantic_segmentation/ai/djl/pytorch/deeplabv3/0.0.1/deeplabv3.zip";
        SemanticSegmentationTranslator translator =
                SemanticSegmentationTranslator.builder()
                        .addTransform(new ToTensor())
                        .addTransform(new Normalize(mean, std))
                        .build();

        Criteria<Image, Segmentation> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.SEMANTIC_SEGMENTATION)
                        .setTypes(Image.class, Segmentation.class)
                        .optModelUrls(url)
                        .optTranslator(translator)
                        .optEngine("PyTorch")
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<Image, Segmentation> model = criteria.loadModel();
                Predictor<Image, Segmentation> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }
}
