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
package ai.djl.examples.inference.face;

import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class FeatureExtractionTest {

    @Test
    public void testFeatureComparison() throws ModelException, TranslateException, IOException {
        Path imageFile = Paths.get("src/test/resources/kana1.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);
        float[] feature = FeatureExtraction.predict(img);
        Assert.assertEquals(feature.length, 512);
        float[] expected = {
            -0.040261813f,
            -0.019486334f,
            -0.09802657f,
            0.017009983f,
            0.037828982f,
            0.030801114f,
            -0.02714689f,
            0.042024296f,
            -0.009838469f,
            -0.005961003f
        };
        float[] sub = new float[10];
        System.arraycopy(feature, 0, sub, 0, 10);
        Assert.assertEquals(sub, expected, 0.0001f);
    }
}
