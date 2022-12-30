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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

public class ImageTranslator implements NoBatchifyTranslator<Image, float[]> {

    /** {@inheritDoc} */
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray array = list.singletonOrThrow();
        return array.toFloatArray();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);

        float percent = 224f / Math.min(input.getWidth(), input.getHeight());
        int resizedWidth = Math.round(input.getWidth() * percent);
        int resizedHeight = Math.round(input.getHeight() * percent);

        array =
                NDImageUtils.resize(
                        array, resizedWidth, resizedHeight, Image.Interpolation.BICUBIC);
        array = NDImageUtils.centerCrop(array, 224, 224);
        array = NDImageUtils.toTensor(array);
        NDArray placeholder = ctx.getNDManager().create("");
        placeholder.setName("module_method:get_image_features");
        return new NDList(array.expandDims(0), placeholder);
    }
}
