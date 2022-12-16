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

import ai.djl.ModelException;
import ai.djl.examples.inference.clip.ClipModel;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class ClipModelTest {

    @Test
    public void testCLIPFeature() throws ModelException, IOException, TranslateException {
        TestRequirements.nightly();
        TestRequirements.engine("PyTorch");

        ClipModel model = new ClipModel();
        String input = "This is a nice day";
        float[] textVector = model.extractTextFeatures(input);
        Image img =
                ImageFactory.getInstance()
                        .fromUrl("http://images.cocodataset.org/val2017/000000039769.jpg");
        float[] imgVector = model.extractImageFeatures(img);
        Assert.assertEquals(textVector.length, imgVector.length);
    }

    @Test
    public void testCLIPComparison() throws ModelException, IOException, TranslateException {
        TestRequirements.nightly();
        TestRequirements.engine("PyTorch");

        ClipModel model = new ClipModel();
        Image img =
                ImageFactory.getInstance()
                        .fromUrl("http://images.cocodataset.org/val2017/000000039769.jpg");
        String text = "A photo of cats";
        String text2 = "A photo of dogs";
        float[] logit0 = model.compareTextAndImage(img, text);
        float[] logit1 = model.compareTextAndImage(img, text2);
        Assert.assertTrue(logit0[0] > logit1[0]);
    }
}
