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
package ai.djl.tflite.engine;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TfLiteTest {

    @Test
    void testTflite()
            throws IOException, MalformedModelException, TranslateException,
                    ModelNotFoundException {
        if (System.getProperty("os.name").toLowerCase().startsWith("win")) {
            throw new SkipException("test only work on mac and Linux");
        }
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optEngine("TFLite")
                        .optFilter("dataset", "aiyDish")
                        .build();
        try (ZooModel<Image, Classifications> model = criteria.loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Image image =
                    ImageFactory.getInstance()
                            .fromUrl("https://resources.djl.ai/images/sachertorte.jpg");
            Classifications prediction = predictor.predict(image);
            Assert.assertEquals(prediction.best().getClassName(), "Sachertorte");
        }
    }
}
