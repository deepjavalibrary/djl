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
package ai.djl.dlr.integration;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class DlrTest {

    @Test
    public void testDlr() throws ModelException, IOException, TranslateException {
        String os;
        if (System.getProperty("os.name").toLowerCase().startsWith("mac")) {
            os = "osx";
        } else if (System.getProperty("os.name").toLowerCase().startsWith("linux")) {
            os = "linux";
        } else {
            throw new SkipException("test only work on mac and Linux");
        }
        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(224, 224))
                        .addTransform(new ToTensor())
                        .build();
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .optFilter("layers", "50")
                        .optFilter("os", os)
                        .optTranslator(translator)
                        .optEngine("DLR")
                        .optProgress(new ProgressBar())
                        .build();
        Path file = Paths.get("../../examples/src/test/resources/kitten.jpg");
        Image image = ImageFactory.getInstance().fromFile(file);
        try (ZooModel<Image, Classifications> model = criteria.loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Classifications result = predictor.predict(image);
            Assert.assertEquals(result.best().getClassName(), "n02123045 tabby, tabby cat");
        }
    }
}
