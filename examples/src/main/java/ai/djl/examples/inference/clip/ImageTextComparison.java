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
        String text = "A photo of cats";
        String text2 = "A photo of dogs";
        double[] probs = compareTextAndImage(text, text2);
        logger.info("{} Probability: {}", text, probs[0]);
        logger.info("{} Probability: {}", text2, probs[1]);
    }

    static double[] compareTextAndImage(String text, String text2)
            throws ModelException, IOException, TranslateException {
        try (ClipModel model = new ClipModel()) {
            String url = "https://resources.djl.ai/images/000000039769.jpg";
            Image img = ImageFactory.getInstance().fromUrl(url);
            float[] logit0 = model.compareTextAndImage(img, text);
            float[] logit1 = model.compareTextAndImage(img, text2);
            double total = Arrays.stream(new double[] {logit0[0], logit1[0]}).map(Math::exp).sum();
            return new double[] {Math.exp(logit0[0]) / total, Math.exp(logit1[0]) / total};
        }
    }
}
