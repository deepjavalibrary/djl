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
package org.apache.mxnet.zoo.cv.actionrecognition;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import software.amazon.ai.Model;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.modality.cv.ImageTranslator;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.modality.cv.util.NDImageUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Utils;

public class ActionRecognitionTranslator extends ImageTranslator<Classification> {

    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        // 299 is the minimum length for inception, 224 for vgg
        BufferedImage cropped = BufferedImageUtils.centerCrop(input, 299, 299);
        return super.processInput(ctx, cropped);
    }

    @Override
    public Classification processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();
        NDArray probabilitiesNd = list.head().get(0);
        List<String> synset = model.getArtifact("classes.txt", Utils::readLines);
        return new Classification(synset, probabilitiesNd);
    }

    @Override
    protected NDArray normalize(NDArray array) {
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        return NDImageUtils.normalize(array, mean, std);
    }
}
