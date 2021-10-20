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
package ai.djl.paddlepaddle.zoo.cv.imageclassification;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import java.util.Arrays;
import java.util.List;

/** A {@link PpWordRotateTranslator} that classify the words and rotate 90 degree if necessary. */
public class PpWordRotateTranslator implements NoBatchifyTranslator<Image, Classifications> {

    List<String> classes;

    /** The Translator for {@link PpWordRotateTranslator}. */
    public PpWordRotateTranslator() {
        classes = Arrays.asList("No Rotate", "Rotate");
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray prob = list.singletonOrThrow();
        return new Classifications(classes, prob);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray img = input.toNDArray(ctx.getNDManager());
        int[] hw = resize32(input.getHeight(), input.getWidth());
        img = NDImageUtils.resize(img, hw[1], hw[0]);
        img = NDImageUtils.toTensor(img).sub(0.5f).div(0.5f);
        img = img.expandDims(0);
        return new NDList(img);
    }

    private int[] resize32(double h, double w) {
        double min = Math.min(h, w);
        if (min < 32) {
            h = 32.0 / min * h;
            w = 32.0 / min * w;
        }
        int h32 = (int) h / 32;
        int w32 = (int) w / 32;
        return new int[] {h32 * 32, w32 * 32};
    }
}
