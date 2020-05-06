/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorflow.integration.modality.cv;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class ResNetTest {

    @Test
    public void testResNet50V1() throws IOException, ModelException, TranslateException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Tensorflow doesn't support Windows yet.");
        }

        Criteria<BufferedImage, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(BufferedImage.class, Classifications.class)
                        .optArtifactId("resnet")
                        .optFilter("layers", "50")
                        .optFilter("flavor", "v1")
                        .optProgress(new ProgressBar())
                        .build();

        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(224, 224))
                .add(
                        new Normalize(
                                new float[] {103.939f, 116.779f, 123.68f},
                                new float[] {1f, 1f, 1f}));

        ImageClassificationTranslator myTranslator =
                ImageClassificationTranslator.builder()
                        .setPipeline(pipeline)
                        .setSynsetArtifactName("synset.txt")
                        .optApplySoftmax(false)
                        .build();
        try (ZooModel<BufferedImage, Classifications> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, Classifications> predictor =
                    model.newPredictor(myTranslator)) {
                Classifications result =
                        predictor.predict(
                                BufferedImageUtils.fromFile(
                                        Paths.get("../../examples/src/test/resources/kitten.jpg")));
                System.out.println(result.best().getClassName());
                Assert.assertTrue(result.best().getClassName().equals("n02124075 Egyptian cat"));
            }
        }
    }
}
