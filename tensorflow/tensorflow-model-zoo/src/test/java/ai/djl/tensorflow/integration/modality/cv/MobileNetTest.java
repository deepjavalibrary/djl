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
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class MobileNetTest {

    @Test
    public void testMobileNetV2() throws IOException, ModelException, TranslateException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Tensorflow doesn't support Windows yet.");
        }

        ImageClassificationTranslator myTranslator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(224, 224))
                        .addTransform(array -> array.div(127.5f).sub(1f))
                        .build();
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optTranslator(myTranslator)
                        .optArtifactId("mobilenet")
                        .optFilter("flavor", "v2")
                        .optProgress(new ProgressBar())
                        .build();

        Path file = Paths.get("../../examples/src/test/resources/kitten.jpg");
        Image img = ImageFactory.getInstance().fromFile(file);
        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Classifications result = predictor.predict(img);
            Assert.assertEquals(result.best().getClassName(), "n02124075 Egyptian cat");
        }
    }
}
