/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class FeatureComparison {

    private static final Logger logger = LoggerFactory.getLogger(FeatureComparison.class);

    private FeatureComparison() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            logger.info("This example only works for PyTorch.");
            return;
        }

        Path imageFile1 = Paths.get("src/test/resources/kana1.jpg");
        Image img1 = ImageFactory.getInstance().fromFile(imageFile1);
        Path imageFile2 = Paths.get("src/test/resources/kana2.jpg");
        Image img2 = ImageFactory.getInstance().fromFile(imageFile2);

        float[] feature1 = FeatureExtraction.predict(img1);
        float[] feature2 = FeatureExtraction.predict(img2);

        logger.info(Float.toString(calculSimilar(feature1, feature2)));
    }

    public static float calculSimilar(float[] feature1, float[] feature2) {
        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        return (float) ((ret / Math.sqrt(mod1) / Math.sqrt(mod2) + 1) / 2.0f);
    }
}
