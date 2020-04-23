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
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MobileNetTest {

    @Test
    public void testMobileNetV2() throws IOException, ModelException, TranslateException {
        Criteria<BufferedImage, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(BufferedImage.class, Classifications.class)
                        .optArtifactId("mobilenet")
                        .optFilter("flavor", "v2")
                        .optProgress(new ProgressBar())
                        .build();

        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(224, 224)).add(new CustomTransform());

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
                Assert.assertTrue(result.best().getClassName().equals("n02124075 Egyptian cat"));
            }
        }
    }

    private static final class CustomTransform implements Transform {

        /** {@inheritDoc} */
        @Override
        public NDArray transform(NDArray array) {
            return array.div(127.5f).sub(1f);
        }
    }
}
