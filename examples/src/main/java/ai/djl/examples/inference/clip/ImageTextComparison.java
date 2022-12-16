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
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

public final class ImageTextComparison {

    private static final Logger logger = LoggerFactory.getLogger(ImageTextComparison.class);

    private ImageTextComparison() {}

    public static void main(String[] args) throws ModelException, IOException, TranslateException {
        ClipModel model = new ClipModel();
        Image img =
                ImageFactory.getInstance()
                        .fromUrl("http://images.cocodataset.org/val2017/000000039769.jpg");
        String text = "A photo of cats";
        String text2 = "A photo of dogs";
        float[] logit0 = model.compareTextAndImage(img, text);
        float[] logit1 = model.compareTextAndImage(img, text2);
        double total = Arrays.stream(new double[] {logit0[0], logit1[0]}).map(Math::exp).sum();
        logger.info("{} Probability: {}", text, Math.exp(logit0[0]) / total);
        logger.info("{} Probability: {}", text2, Math.exp(logit1[0]) / total);
    }
}
