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
package ai.djl.examples.inference.clip;

import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class ClipModelTest {

    @Test
    public void testClipFeature() throws ModelException, IOException, TranslateException {
        TestRequirements.linux();
        TestRequirements.nightly();

        try (ClipModel model = new ClipModel()) {
            String input = "This is a nice day";
            String url = "http://images.cocodataset.org/val2017/000000039769.jpg";
            float[] textVector = model.extractTextFeatures(input);
            Image img = ImageFactory.getInstance().fromUrl(url);
            float[] imgVector = model.extractImageFeatures(img);
            Assert.assertEquals(textVector.length, imgVector.length);
            assertAlmostEquals(textVector[0], 0.09463542);
            assertAlmostEquals(imgVector[0], -0.12755919);
        }
    }

    @Test
    public void testClipComparison() throws ModelException, IOException, TranslateException {
        TestRequirements.linux();
        TestRequirements.nightly();

        String text = "A photo of cats";
        String text2 = "A photo of dogs";
        double[] probs = ImageTextComparison.compareTextAndImage(text, text2);
        Assert.assertTrue(probs[0] > 0.9);
        Assert.assertTrue(probs[1] < 0.1);
    }

    public static void assertAlmostEquals(double actual, double expected) {
        Assert.assertTrue(Math.abs(actual - expected) < 1e-3 + 1e-5 * Math.abs(expected));
    }
}
