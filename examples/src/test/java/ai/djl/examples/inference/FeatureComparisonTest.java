/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.examples.inference.face.FeatureComparison;
import ai.djl.examples.inference.face.FeatureExtraction;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class FeatureComparisonTest {

    @Test
    public void testFeatureComparison() throws ModelException, TranslateException, IOException {
        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            throw new SkipException("Only works for PyTorch engine.");
        }

        if (Boolean.getBoolean("nightly")) {
            Path imageFile1 = Paths.get("src/test/resources/kana1.jpg");
            Image img1 = ImageFactory.getInstance().fromFile(imageFile1);
            Path imageFile2 = Paths.get("src/test/resources/kana2.jpg");
            Image img2 = ImageFactory.getInstance().fromFile(imageFile2);

            float[] feature1 = FeatureExtraction.predict(img1);
            float[] feature2 = FeatureExtraction.predict(img2);

            Assert.assertTrue(
                    Double.compare(FeatureComparison.calculSimilar(feature1, feature2), 0.6) > 0);
        }
    }
}
