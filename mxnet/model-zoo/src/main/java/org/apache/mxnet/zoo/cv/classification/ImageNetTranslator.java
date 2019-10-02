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

package org.apache.mxnet.zoo.cv.classification;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import software.amazon.ai.Model;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.modality.cv.ImageTranslator;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Utils;

public class ImageNetTranslator extends ImageTranslator<Classification> {

    private int imageWidth = 224;
    private int imageHeight = 224;

    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        BufferedImage image = BufferedImageUtils.centerCrop(input);
        image = BufferedImageUtils.resize(image, imageWidth, imageHeight);

        return super.processInput(ctx, image);
    }

    @Override
    public Classification processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();

        NDArray probabilitiesNd = list.head().get(0);
        List<String> synset = model.getArtifact("synset.txt", Utils::readLines);
        return new Classification(synset, probabilitiesNd);
    }
}
